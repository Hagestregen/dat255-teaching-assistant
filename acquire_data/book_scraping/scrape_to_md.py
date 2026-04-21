#!/usr/bin/env python3
"""
scrape_to_md.py
Usage: python scrape_to_md.py <url> [output_file.md] [--source "Book Name"]

Extracts the main text content from a webpage and saves it as Markdown.
Writes YAML frontmatter with a `source:` field so the chunker can build
multi-source breadcrumbs (Source > Chapter > Section).

Source detection order:
  1. --source flag (explicit override, always wins)
  2. JSON-LD isPartOf.name  (e.g. "Deep Learning with Python")
  3. OG/meta title as last resort

Chapter title detection order:
  1. JSON-LD name field  (e.g. "Chapter 1 - What is deep learning?")
  2. Structural title div

Output heading hierarchy per file:
    # Chapter N: Chapter title       <- H1
    ## Section heading               <- H2
    ### Subsection heading           <- H3

Requirements:
    pip install requests trafilatura
"""

import argparse
import json
import re
import sys
import requests
import trafilatura
from trafilatura.settings import use_config
from urllib.parse import urlparse
from pathlib import Path


def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    slug   = parsed.path.strip("/").replace("/", "_") or parsed.hostname
    slug   = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug)
    slug   = re.sub(r"_+", "_", slug).strip("_")
    return (slug[:80] or "output") + ".md"


def _parse_jsonld(html: str) -> dict:
    match = re.search(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.DOTALL | re.IGNORECASE,
    )
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def _extract_source_name(html: str) -> str | None:
    """Detect the source (book/series) name. Returns None if nothing found."""
    data = _parse_jsonld(html)

    # schema.org Chapter/Article: isPartOf.name = book title
    is_part_of = data.get("isPartOf", {})
    if isinstance(is_part_of, dict):
        name = is_part_of.get("name", "").strip()
        if name:
            return name

    # Top-level name only if it doesn't look like a chapter title
    top_name = data.get("name", "").strip()
    if top_name and not re.match(r'^chapter\s+\d+', top_name, re.IGNORECASE):
        return top_name

    return None


def _extract_chapter_info(html: str) -> tuple[str | None, str | None]:
    """Return (chapter_label, chapter_title), e.g. ("Chapter 1", "What is deep learning?")"""
    data = _parse_jsonld(html)
    name = data.get("name", "")
    m    = re.match(r'^(Chapter\s+\d+)\s*[-:]\s*(.+)$', name, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    div_match = re.search(
        r'<div[^>]+class=["\'][^"\']*\btitle\b[^"\']*["\'][^>]*>'
        r'.*?<p[^>]+class=["\'][^"\']*chapter-name[^"\']*["\'][^>]*>([^<]+)</p>'
        r'\s*<h1[^>]*>([^<]+)</h1>',
        html, re.DOTALL | re.IGNORECASE,
    )
    if div_match:
        return div_match.group(1).strip(), div_match.group(2).strip()

    return None, None


def _strip_spurious_h1(markdown: str) -> str:
    """Remove a leading H1 that trafilatura may have pulled from the page."""
    lines = markdown.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("# "):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines)


def _build_frontmatter(source: str) -> str:
    safe = source.replace('"', '\\"')
    return f'---\nsource: "{safe}"\n---\n\n'


def scrape(url: str, output_path: str | None = None, source_override: str | None = None) -> Path:
    headers  = {"User-Agent": "Mozilla/5.0 (compatible; scraper/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    html = response.text

    config = use_config()
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

    markdown = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        config=config,
    )

    if not markdown:
        raise ValueError(
            "Could not extract any content. "
            "The page may require JavaScript or block scraping."
        )

    markdown = _strip_spurious_h1(markdown)

    # Resolve source name
    source = source_override or _extract_source_name(html)
    if source:
        print(f"  Source: {source}")
        frontmatter = _build_frontmatter(source)
    else:
        frontmatter = ""
        print("  [INFO] No source detected. Pass --source if you need multi-source tracking.")

    # Build chapter H1
    chapter_label, chapter_title = _extract_chapter_info(html)
    if chapter_label and chapter_title:
        heading = f"# {chapter_label}: {chapter_title}\n\n"
        print(f"  Chapter: {chapter_label}: {chapter_title}")
    elif chapter_title:
        heading = f"# {chapter_title}\n\n"
        print(f"  Chapter: {chapter_title}")
    else:
        meta     = trafilatura.extract_metadata(html)
        fallback = meta.title if (meta and meta.title) else url
        heading  = f"# {fallback}\n\n"
        print(f"  [WARN] Could not extract chapter title; using: {fallback!r}")

    if output_path is None:
        output_path = url_to_filename(url)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(frontmatter + heading + markdown, encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="Scrape a webpage to structured Markdown.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url",    help="Page URL to scrape")
    parser.add_argument("output", nargs="?", default=None, help="Output .md path")
    parser.add_argument("--source", default=None, metavar="NAME",
                        help="Source/book name for frontmatter. Auto-detected from JSON-LD if omitted.")
    args = parser.parse_args()

    print(f"Fetching: {args.url}")
    path = scrape(args.url, args.output, source_override=args.source)
    print(f"Saved to: {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()