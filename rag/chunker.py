#!/usr/bin/env python3
"""
chunker.py — structure-aware two-pass markdown chunker
==========
Reads a manifest .txt file listing folders and/or .md files, applies
structure-aware two-pass chunking, and writes all chunks to chunks.json.

Pass 1 — Logical split:  MarkdownHeaderTextSplitter breaks on #/##/###/####
Pass 2 — Size cap:       RecursiveCharacterTextSplitter sub-chunks anything
                         still too large, preserving header metadata.
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

HEADERS_TO_SPLIT_ON = [
    ("#",    "h1"),
    ("##",   "h2"),
    ("###",  "h3"),
    ("####", "h4"),
]

# Sized to match what quiz.py actually feeds the model.
# Overlap is ~15% of chunk size — enough to avoid cutting mid-concept.
PASS2_CHUNK_SIZE    = 600
PASS2_CHUNK_OVERLAP = 80

# Chunks shorter than this are almost always orphaned headers or
# one-line stubs — not useful for retrieval or question generation.
MIN_CHUNK_CHARS = 80

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False,
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PASS2_CHUNK_SIZE,
    chunk_overlap=PASS2_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_manifest(manifest_path: str) -> list[Path]:
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
    parts = []
    if source:
        parts.append(source)
    for key in ("h1", "h2", "h3", "h4"):
        val = metadata.get(key)
        if val:
            parts.append(val.strip())
    return " > ".join(parts)


def _is_stub(text: str) -> bool:
    """
    True if the chunk is just a header line with no real body content.
    e.g. "# Chapter 1: What is deep learning?" with nothing below it.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return True
    # All lines are markdown headers → no body content
    if all(l.startswith("#") for l in lines):
        return True
    # Too short to be useful
    if len(text.strip()) < MIN_CHUNK_CHARS:
        return True
    return False


def chunk_file(md_path: Path) -> list[dict]:
    """
    Two-pass chunk a single .md file.

    The breadcrumb is stored in metadata only — it is NOT baked into
    the text field. quiz.py can prepend it as a clearly-labelled header
    at generation time, keeping it out of the model's question output.
    """
    raw = md_path.read_text(encoding="utf-8")
    frontmatter, content = _parse_frontmatter(raw)
    source_name          = frontmatter.get("source") or None

    header_docs = markdown_splitter.split_text(content)
    final_docs  = text_splitter.split_documents(header_docs)

    chunks: list[dict] = []
    for doc in final_docs:
        text = doc.page_content.strip()
        if _is_stub(text):
            continue

        meta = {
            "h1":          doc.metadata.get("h1"),
            "h2":          doc.metadata.get("h2"),
            "h3":          doc.metadata.get("h3"),
            "h4":          doc.metadata.get("h4"),
            "source_name": source_name,
            "breadcrumb":  build_breadcrumb(doc.metadata, source_name),
        }

        # Clean text: strip any leading markdown header lines that are
        # just the section title repeated (common after the splitter).
        clean_text = re.sub(r'^#+\s+.+\n', '', text, count=1).strip()
        if not clean_text:
            continue

        chunks.append({
            "text":     clean_text,    # pure content, no metadata prefix
            "source":   str(md_path),
            "metadata": meta,
        })

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Structure-aware Markdown chunker")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output",   default="rag_index/chunks.json")
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