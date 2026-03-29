#!/usr/bin/env python3
"""
scrape_slides.py  (v2 — fragment deduplication + cleaner formatting)
=====================================================================
Scrapes all RevealJS/Quarto slide decks from the DAT255 course website
and converts each into a clean Markdown file for RAG / LLM ingestion.

Usage:
    python scrape_slides.py

Output:
    One .md file per slide deck, saved to ../../data/slide_md/

Requirements:
    pip install requests beautifulsoup4
"""

import re
import sys
import time
import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_URL  = "https://hvl-ml.github.io/DAT255/"
BASE_URL   = "https://hvl-ml.github.io/DAT255"
OUTPUT_DIR = Path("../../data/slide_md")
SLIDE_RE   = re.compile(r"\./slides/(\d{2}-[^\"]+\.html)")
# ──────────────────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; dat255-scraper/1.0)"}


# ── Index parsing ─────────────────────────────────────────────────────────────

def discover_slides() -> list[tuple[str, str]]:
    resp = requests.get(INDEX_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.find_all("a", href=SLIDE_RE):
        href  = a["href"]
        title = a.get_text(strip=True)
        url   = BASE_URL + "/" + href.lstrip("./")
        results.append((title, url))
    return results


# ── HTML → Markdown ───────────────────────────────────────────────────────────

def tag_to_md(tag, depth: int = 0) -> str:
    if isinstance(tag, NavigableString):
        return str(tag)

    name = tag.name if tag.name else ""

    if name in ("script", "style", "link", "meta", "head",
                "button", "nav", "footer", "svg", "figure"):
        return ""

    if name == "img":
        alt = tag.get("alt", "").strip()
        return f"*[image: {alt}]*" if alt else ""

    if name in ("h1", "h2", "h3", "h4"):
        inner = children_to_md(tag, depth).strip()
        if not inner:
            return ""
        level = int(name[1])
        return f"\n{'#' * level} {inner}\n"

    if name == "p":
        inner = children_to_md(tag, depth).strip()
        return f"\n{inner}\n" if inner else ""

    if name == "ul":
        items = []
        for li in tag.find_all("li", recursive=False):
            text = children_to_md(li, depth + 1).strip()
            text = re.sub(r"\s*\n\s*", " ", text).strip()
            if text:
                items.append("  " * depth + f"- {text}")
        return ("\n" + "\n".join(items) + "\n") if items else ""

    if name == "ol":
        items = []
        for i, li in enumerate(tag.find_all("li", recursive=False), 1):
            text = children_to_md(li, depth + 1).strip()
            text = re.sub(r"\s*\n\s*", " ", text).strip()
            if text:
                items.append("  " * depth + f"{i}. {text}")
        return ("\n" + "\n".join(items) + "\n") if items else ""

    if name == "code":
        if tag.parent and tag.parent.name == "pre":
            return tag.get_text()
        return f"`{tag.get_text()}`"

    if name == "pre":
        code_tag = tag.find("code")
        lang = ""
        if code_tag:
            cls = " ".join(code_tag.get("class", []))
            m = re.search(r"language-(\w+)", cls)
            if m:
                lang = m.group(1)
            code = code_tag.get_text().strip()
        else:
            code = tag.get_text().strip()
        return f"\n```{lang}\n{code}\n```\n"

    if name == "em":
        inner = children_to_md(tag, depth).strip()
        return f"*{inner}*" if inner else ""

    if name in ("strong", "b"):
        inner = children_to_md(tag, depth).strip()
        return f"**{inner}**" if inner else ""

    if name == "blockquote":
        inner = children_to_md(tag, depth).strip()
        return f"\n> {inner.replace(chr(10), chr(10) + '> ')}\n"

    if name == "hr":
        return "\n---\n"

    if name == "br":
        return "\n"

    if name == "span":
        cls = " ".join(tag.get("class", []))
        if "math" in cls:
            return tag.get_text()
        return children_to_md(tag, depth)

    return children_to_md(tag, depth)


def children_to_md(tag, depth: int = 0) -> str:
    return "".join(tag_to_md(child, depth) for child in tag.children)


# ── Slide extraction ──────────────────────────────────────────────────────────

def section_to_md(section: Tag) -> str | None:
    """
    Convert one <section> to Markdown text. Returns None for skippable slides.
    """
    # Interactive iframes (TF playground, etc.)
    if section.get("data-background-iframe"):
        menu_title = section.get("data-menu-title", "interactive demo")
        return f"*[Interactive slide: {menu_title}]*"

    md = children_to_md(section).strip()
    return md if md else None


# ── Fragment deduplication ────────────────────────────────────────────────────

def _first_heading(md: str) -> str | None:
    """Return the first Markdown heading text found, or None."""
    for line in md.splitlines():
        m = re.match(r"^#{1,4}\s+(.+)", line.strip())
        if m:
            return m.group(1).strip()
    return None


def deduplicate_fragments(slides: list[str]) -> list[str]:
    """
    RevealJS 'fragment' animations repeat the same slide multiple times,
    adding a bit more content each time. When consecutive slides share the
    same heading, keep only the LAST one (the most complete version).
    """
    if not slides:
        return slides

    result = []
    i = 0
    while i < len(slides):
        current_heading = _first_heading(slides[i])
        if current_heading:
            # Find how far this same-headed run extends
            j = i + 1
            while j < len(slides) and _first_heading(slides[j]) == current_heading:
                j += 1
            result.append(slides[j - 1])   # keep the last (most complete)
            i = j
        else:
            result.append(slides[i])
            i += 1
    return result


# ── Full deck conversion ──────────────────────────────────────────────────────

def fetch_slide_deck(url: str, index_title: str, stem: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    page_title_tag = soup.find("title")
    page_title = page_title_tag.get_text(strip=True) if page_title_tag else index_title

    slides_div = soup.find("div", class_="slides")
    if not slides_div:
        return f"# {page_title}\n\n*Could not find slide content.*\n"

    # Collect all individual slide sections (handle nested vertical stacks)
    raw_slides: list[str] = []
    for top in slides_div.find_all("section", recursive=False):
        nested = top.find_all("section", recursive=False)
        if nested:
            for sub in nested:
                md = section_to_md(sub)
                if md:
                    raw_slides.append(md)
        else:
            md = section_to_md(top)
            if md:
                raw_slides.append(md)

    deduped = deduplicate_fragments(raw_slides)

    header = (
        f"<!-- source: {stem}.html -->\n"
        f"<!-- index-title: {index_title} -->\n\n"
        f"# {page_title}\n"
    )

    return header + "\n\n---\n\n".join(deduped) + "\n"


# ── Post-processing ───────────────────────────────────────────────────────────

def clean_markdown(md: str) -> str:
    md = md.replace("\r\n", "\n")
    # Collapse 3+ blank lines → 1 blank line
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Strip trailing whitespace
    md = "\n".join(line.rstrip() for line in md.splitlines())
    return md.strip() + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Fetching slide index …")
    try:
        slides = discover_slides()
    except requests.HTTPError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not slides:
        print("No slides found — check SLIDE_RE pattern or INDEX_URL.")
        sys.exit(1)

    print(f"Found {len(slides)} slide decks.\n")

    for index_title, url in slides:
        slug = re.search(r"/slides/(.+?)\.html", url)
        stem = slug.group(1) if slug else url.split("/")[-1].replace(".html", "")
        output_path = OUTPUT_DIR / f"{stem}.md"

        print(f"  ↓  {stem}.html  [{index_title}]")
        try:
            md = fetch_slide_deck(url, index_title, stem)
            md = clean_markdown(md)
        except Exception as e:
            print(f"     WARN: could not process — {e}")
            continue

        output_path.write_text(md, encoding="utf-8")
        print(f"     → {output_path}  ({len(md):,} chars)")
        time.sleep(0.4)

    print(f"\nDone! Markdown files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()