#!/usr/bin/env python3
"""
chunker.py
==========
Reads a manifest .txt file listing folders and/or .md files, applies
structure-aware two-pass chunking, and writes all chunks to chunks.json.

Pass 1 — Logical split:  MarkdownHeaderTextSplitter breaks on #/##/###/####
Pass 2 — Size cap:       RecursiveCharacterTextSplitter sub-chunks anything
                         still too large, preserving header metadata.

Usage:
    python chunker.py --manifest sources.txt [--output chunks.json]

Manifest format (sources.txt):
    # Lines starting with # are comments and are ignored
    /path/to/folder_of_md_files
    /path/to/single_file.md
    relative/paths/work/too

Requirements:
    pip install langchain langchain-text-splitters
"""

import argparse
import json
import re
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ── Chunking config ───────────────────────────────────────────────────────────
# Headers that act as hard logical boundaries
HEADERS_TO_SPLIT_ON = [
    ("#",    "h1"),
    ("##",   "h2"),
    ("###",  "h3"),
    ("####", "h4"),
]

# A section larger than this (chars) gets sub-chunked in Pass 2
PASS2_CHUNK_SIZE    = 1000   # ~250 tokens for most embedding models
PASS2_CHUNK_OVERLAP = 100    # small overlap so sentences at boundaries aren't lost
# ──────────────────────────────────────────────────────────────────────────────

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False,   # keep the heading text inside the chunk content too
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PASS2_CHUNK_SIZE,
    chunk_overlap=PASS2_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Manifest loading ──────────────────────────────────────────────────────────

def load_manifest(manifest_path: str) -> list[Path]:
    """
    Parse the manifest file and return a flat list of .md file Paths.
    Folders are expanded to all their .md files (non-recursive by default,
    but you can change glob below).
    """
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    md_files: list[Path] = []
    seen: set[Path] = set()

    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue   # skip empty lines and comments

        p = Path(line)
        if not p.exists():
            print(f"  [WARN] Path not found, skipping: {p}")
            continue

        if p.is_dir():
            # Recursively find all .md files inside this folder
            found = sorted(p.rglob("*.md"))
            if not found:
                print(f"  [WARN] No .md files found in: {p}")
            for f in found:
                if f not in seen:
                    md_files.append(f)
                    seen.add(f)
        elif p.suffix.lower() == ".md":
            if p not in seen:
                md_files.append(p)
                seen.add(p)
        else:
            print(f"  [WARN] Not a .md file or directory, skipping: {p}")

    return md_files


# ── Chunking ──────────────────────────────────────────────────────────────────

def build_breadcrumb(metadata: dict) -> str:
    """
    Turn {h1: 'Intro', h2: 'Self-Attention'} into 'Intro > Self-Attention'.
    Used to make each chunk semantically self-contained.
    """
    parts = []
    for key in ("h1", "h2", "h3", "h4"):
        if key in metadata and metadata[key]:
            parts.append(metadata[key].strip())
    return " > ".join(parts)


def chunk_file(md_path: Path) -> list[dict]:
    """
    Two-pass chunk a single .md file.
    Returns a list of dicts:
        {
          "text":     str,        # chunk content (with breadcrumb prepended)
          "source":   str,        # relative file path
          "metadata": {
              "h1": str | None,
              "h2": str | None,
              "h3": str | None,
              "h4": str | None,
              "breadcrumb": str,
          }
        }
    """
    raw = md_path.read_text(encoding="utf-8")
    source = str(md_path)

    # ── Pass 1: split on headers ──────────────────────────────────────────────
    header_docs = markdown_splitter.split_text(raw)

    # ── Pass 2: sub-chunk oversized sections ──────────────────────────────────
    final_docs = text_splitter.split_documents(header_docs)

    chunks: list[dict] = []
    for doc in final_docs:
        text = doc.page_content.strip()
        if not text:
            continue

        meta = {
            "h1": doc.metadata.get("h1"),
            "h2": doc.metadata.get("h2"),
            "h3": doc.metadata.get("h3"),
            "h4": doc.metadata.get("h4"),
            "breadcrumb": build_breadcrumb(doc.metadata),
        }

        # Prepend the breadcrumb so it's retrievable even out of context
        if meta["breadcrumb"]:
            enriched_text = f"[{meta['breadcrumb']}]\n{text}"
        else:
            enriched_text = text

        chunks.append({
            "text":     enriched_text,
            "source":   source,
            "metadata": meta,
        })

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Structure-aware Markdown chunker")
    parser.add_argument("--manifest", required=True,
                        help="Path to the manifest .txt file")
    parser.add_argument("--output",   default="chunks.json",
                        help="Output JSON file (default: chunks.json)")
    args = parser.parse_args()

    print(f"Loading manifest: {args.manifest}")
    md_files = load_manifest(args.manifest)
    print(f"Found {len(md_files)} markdown file(s) to process.\n")

    all_chunks: list[dict] = []
    for md_path in md_files:
        print(f"  Chunking: {md_path}")
        try:
            chunks = chunk_file(md_path)
            all_chunks.extend(chunks)
            print(f"            → {len(chunks)} chunks")
        except Exception as e:
            print(f"  [ERROR] {md_path}: {e}")

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to:     {output_path}")


if __name__ == "__main__":
    main()