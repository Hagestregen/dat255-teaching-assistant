#!/usr/bin/env python3
"""
chunker.py
==========
Reads a manifest .txt file listing folders and/or .md files, applies
structure-aware two-pass chunking, and writes all chunks to chunks.json.

Pass 1 — Logical split:  MarkdownHeaderTextSplitter breaks on #/##/###/####
Pass 2 — Size cap:       RecursiveCharacterTextSplitter sub-chunks anything
                         still too large, preserving header metadata.

YAML frontmatter support:
    If a .md file begins with a `---` frontmatter block containing a `source:`
    field, that value is prepended to every chunk's breadcrumb:

        source: "Deep Learning with Python"
        → breadcrumb: "Deep Learning with Python > Chapter 1: ... > Section"

    This is how multiple books/sources are distinguished in the topic tree.
    Files without frontmatter are chunked normally (breadcrumb starts at H1).

Usage:
    python chunker.py --manifest sources.txt [--output rag_index/chunks.json]

Manifest format (sources.txt):
    # Lines starting with # are comments
    /path/to/folder_of_md_files
    /path/to/single_file.md

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

# Chunking config
HEADERS_TO_SPLIT_ON = [
    ("#",    "h1"),
    ("##",   "h2"),
    ("###",  "h3"),
    ("####", "h4"),
]

PASS2_CHUNK_SIZE    = 1000
PASS2_CHUNK_OVERLAP = 100

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PASS2_CHUNK_SIZE,
    chunk_overlap=PASS2_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ","?", "!", " ", ""], # Add "?" and "!" to the separators to split on question marks and exclamation marks
)


def load_manifest(manifest_path: str) -> list[Path]:
    """
    Parse the manifest and return a flat list of .md file Paths.
    Folders are expanded recursively.
    """
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    md_files: list[Path] = []
    seen: set[Path]      = set()

    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        p = Path(line)
        if not p.exists():
            print(f"  [WARN] Path not found, skipping: {p}")
            continue

        if p.is_dir():
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


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Strip and parse a YAML frontmatter block from the top of a markdown string.

    Returns (metadata_dict, remaining_markdown).
    Only the `source:` key is used; other keys are ignored.
    If no frontmatter is present, returns ({}, original_text).
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    block    = text[3:end].strip()
    rest     = text[end + 4:].lstrip("\n")
    metadata = {}

    for line in block.splitlines():
        m = re.match(r'^(\w+)\s*:\s*"?([^"]+)"?\s*$', line)
        if m:
            metadata[m.group(1).strip()] = m.group(2).strip()

    return metadata, rest


def build_breadcrumb(metadata: dict, source: str | None = None) -> str:
    """
    Build a breadcrumb string from chunk metadata.

    If source is provided it is prepended:
        "Deep Learning with Python > Chapter 1: What is DL? > Artificial intelligence"
    Otherwise:
        "Chapter 1: What is DL? > Artificial intelligence"
    """
    parts = []
    if source:
        parts.append(source)
    for key in ("h1", "h2", "h3", "h4"):
        val = metadata.get(key)
        if val:
            parts.append(val.strip())
    return " > ".join(parts)


def chunk_file(md_path: Path) -> list[dict]:
    """
    Two-pass chunk a single .md file.

    Returns a list of dicts:
        {
          "text":     str,
          "source":   str,
          "metadata": {
              "h1": str | None,
              "h2": str | None,
              "h3": str | None,
              "h4": str | None,
              "source_name": str | None,
              "breadcrumb": str,
          }
        }
    """
    raw = md_path.read_text(encoding="utf-8")

    frontmatter, content = _parse_frontmatter(raw)
    source_name          = frontmatter.get("source") or None

    header_docs = markdown_splitter.split_text(content)
    final_docs  = text_splitter.split_documents(header_docs)

    chunks: list[dict] = []
    for doc in final_docs:
        text = doc.page_content.strip()
        if not text:
            continue

        meta = {
            "h1":          doc.metadata.get("h1"),
            "h2":          doc.metadata.get("h2"),
            "h3":          doc.metadata.get("h3"),
            "h4":          doc.metadata.get("h4"),
            "source_name": source_name,
            "breadcrumb":  build_breadcrumb(doc.metadata, source_name),
        }

        prefix       = f"[{meta['breadcrumb']}]\n" if meta["breadcrumb"] else ""
        enriched     = prefix + text

        chunks.append({
            "text":     enriched,
            "source":   str(md_path),
            "metadata": meta,
        })

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Structure-aware Markdown chunker")
    parser.add_argument("--manifest", required=True, help="Path to the manifest .txt file")
    parser.add_argument("--output",   default="rag_index/chunks.json",
                        help="Output JSON file (default: rag_index/chunks.json)")
    args = parser.parse_args()

    print(f"Loading manifest: {args.manifest}")
    md_files = load_manifest(args.manifest)
    print(f"Found {len(md_files)} markdown file(s).\n")

    all_chunks: list[dict] = []
    for md_path in md_files:
        print(f"  Chunking: {md_path}")
        try:
            chunks = chunk_file(md_path)
            all_chunks.extend(chunks)
            print(f"            -> {len(chunks)} chunks")
        except Exception as e:
            print(f"  [ERROR] {md_path}: {e}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to:     {output_path}")


if __name__ == "__main__":
    main()