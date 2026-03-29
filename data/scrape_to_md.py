#!/usr/bin/env python3
"""
scrape_to_md.py
Usage: python scrape_to_md.py <url> [output_file.md]

Extracts the main text content from a webpage (headings + body text,
no nav/sidebar/ads/links) and saves it as a Markdown file.

Requirements:
    pip install requests trafilatura
"""

import sys
import re
import requests
import trafilatura
from trafilatura.settings import use_config
from urllib.parse import urlparse
from pathlib import Path


def url_to_filename(url: str) -> str:
    """Turn a URL into a safe filename if none is provided."""
    parsed = urlparse(url)
    # Use the last path segment, or the hostname if path is empty
    slug = parsed.path.strip("/").replace("/", "_") or parsed.hostname
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return (slug[:80] or "output") + ".md"


def scrape(url: str, output_path: str | None = None) -> Path:
    # --- Fetch ---
    headers = {"User-Agent": "Mozilla/5.0 (compatible; scraper/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    response.encoding = response.apparent_encoding  # fix curly quotes / em-dashes
    html = response.text

    # --- Extract main content as Markdown ---
    config = use_config()
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")  # disable timeout

    markdown = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        no_fallback=False,      # use fallback extractors if needed
        config=config,
    )

    if not markdown:
        raise ValueError("Could not extract any content from the page. "
                         "The page may require JavaScript or block scraping.")

    # --- Optionally prepend the page title ---
    title = trafilatura.extract_metadata(html)
    header = ""
    if title and title.title:
        header = f"# {title.title}\n\n"

    final = header + markdown

    # --- Save ---
    if output_path is None:
        output_path = url_to_filename(url)

    out = Path(output_path)
    out.write_text(final, encoding="utf-8")
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python scrape_to_md.py <url> [output.md]")
        sys.exit(1)

    url = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) >= 3 else None

    print(f"Fetching: {url}")
    path = scrape(url, output)
    print(f"Saved to: {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()