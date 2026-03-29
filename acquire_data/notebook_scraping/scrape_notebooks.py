#!/usr/bin/env python3
"""
scrape_notebooks.py
====================
Fetches all Jupyter notebooks from a public GitHub repo and converts
them to clean Markdown files suitable for RAG / LLM ingestion.

Usage:
    python scrape_notebooks.py

Output:
    One .md file per notebook, saved to ../../data/notebook_md/

Requirements:
    pip install requests
"""

import json
import re
import sys
import time
import os
from pathlib import Path
import requests

# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_API   = "https://api.github.com/repos/{owner}/{repo}/contents/{path}"
RAW_BASE     = "https://raw.githubusercontent.com/{owner}/{repo}/refs/heads/{branch}/{path}"
OWNER        = "HVL-ML"
REPO         = "DAT255"
BRANCH       = "main"
NOTEBOOKS_DIR = "notebooks"
OUTPUT_DIR   = Path("../../data/notebook_md")

# Max length (chars) for a code cell output to be included.
# Longer outputs (matrices, image data, …) are replaced with a placeholder.
MAX_OUTPUT_CHARS = 800
# ──────────────────────────────────────────────────────────────────────────────


def github_api_url(path: str) -> str:
    return GITHUB_API.format(owner=OWNER, repo=REPO, path=path)


def raw_url(path: str) -> str:
    return RAW_BASE.format(owner=OWNER, repo=REPO, branch=BRANCH, path=path)


def list_notebooks() -> list[dict]:
    """Return [{name, path, download_url}, …] for every .ipynb in the repo dir."""
    url = github_api_url(NOTEBOOKS_DIR)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    entries = resp.json()
    notebooks = [e for e in entries if e["name"].endswith(".ipynb")]
    notebooks.sort(key=lambda e: e["name"])
    return notebooks


def fetch_notebook(raw_download_url: str) -> dict:
    resp = requests.get(raw_download_url, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Cell rendering ────────────────────────────────────────────────────────────

def render_markdown_cell(cell: dict) -> str:
    source = "".join(cell.get("source", []))
    return source.strip() + "\n"


def render_output(output: dict) -> str | None:
    """
    Convert a single cell output to a short text snippet, or None to skip.
    Skips image/html outputs; truncates long text.
    """
    otype = output.get("output_type", "")

    # ── stream (print statements) ──
    if otype == "stream":
        text = "".join(output.get("text", []))
        text = text.strip()
        if not text:
            return None
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "\n… [output truncated]"
        return f"```\n{text}\n```"

    # ── execute_result / display_data ──
    if otype in ("execute_result", "display_data"):
        data = output.get("data", {})

        # Skip images and rich HTML
        if "image/png" in data or "image/jpeg" in data:
            return "_[image output — see notebook]_"

        if "text/html" in data:
            # HTML tables etc. — skip for RAG cleanliness
            return "_[HTML output — see notebook]_"

        if "text/plain" in data:
            text = "".join(data["text/plain"]).strip()
            if not text:
                return None
            if len(text) > MAX_OUTPUT_CHARS:
                text = text[:MAX_OUTPUT_CHARS] + "\n… [output truncated]"
            return f"```\n{text}\n```"

    # ── errors ──
    if otype == "error":
        ename  = output.get("ename", "Error")
        evalue = output.get("evalue", "")
        return f"```\n{ename}: {evalue}\n```"

    return None


def render_code_cell(cell: dict) -> str:
    source = "".join(cell.get("source", []))
    parts  = [f"```python\n{source.strip()}\n```"]

    outputs = cell.get("outputs", [])
    rendered_outputs = [render_output(o) for o in outputs]
    rendered_outputs = [r for r in rendered_outputs if r is not None]

    if rendered_outputs:
        parts.append("\n**Output:**\n")
        parts.extend(rendered_outputs)

    return "\n".join(parts) + "\n"


def render_raw_cell(cell: dict) -> str:
    source = "".join(cell.get("source", []))
    if not source.strip():
        return ""
    return f"```\n{source.strip()}\n```\n"


# ── Notebook → Markdown ───────────────────────────────────────────────────────

def notebook_to_markdown(nb: dict, notebook_name: str) -> str:
    """Convert a parsed notebook dict to a Markdown string."""
    lines: list[str] = []

    # File-level header so RAG chunkers know which notebook this came from
    lines.append(f"<!-- source: {notebook_name} -->\n")

    cells = nb.get("cells", [])
    for cell in cells:
        ct = cell.get("cell_type", "")
        if ct == "markdown":
            lines.append(render_markdown_cell(cell))
        elif ct == "code":
            lines.append(render_code_cell(cell))
        elif ct == "raw":
            lines.append(render_raw_cell(cell))

        lines.append("")   # blank line between cells

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Fetching notebook list from GitHub API …")
    try:
        notebooks = list_notebooks()
    except requests.HTTPError as e:
        print(f"ERROR fetching notebook list: {e}")
        sys.exit(1)

    print(f"Found {len(notebooks)} notebooks.\n")

    for entry in notebooks:
        name        = entry["name"]                 # e.g. 01_digit_classification.ipynb
        stem        = Path(name).stem               # e.g. 01_digit_classification
        output_path = OUTPUT_DIR / f"{stem}.md"
        dl_url      = raw_url(f"{NOTEBOOKS_DIR}/{name}")

        print(f"  ↓  {name}")
        try:
            nb = fetch_notebook(dl_url)
        except Exception as e:
            print(f"     WARN: could not fetch — {e}")
            continue

        md = notebook_to_markdown(nb, name)
        output_path.write_text(md, encoding="utf-8")
        print(f"     → {output_path}  ({len(md):,} chars)")

        time.sleep(0.3)   # be polite to GitHub

    print(f"\nDone! Markdown files saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()