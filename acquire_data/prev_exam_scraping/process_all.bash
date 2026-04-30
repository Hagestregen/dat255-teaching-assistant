#!/bin/bash
# process_all.bash
# Runs PDF extraction script using PyMuPDF on every .pdf file in data/previous_exam/
# and writes JSON output to data/previous_exam_extracted_raw/
#
# Usage (from repo root or from this script's folder):
#   bash acquire_data/prev_exam_scraping/process_all.bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

EXAM_DIR="$REPO_ROOT/data/previous_exam"
OUT_DIR="$REPO_ROOT/data/previous_exam_extracted_raw"

SCRIPT_MU="$SCRIPT_DIR/extract_pymupdf.py"

mkdir -p "$OUT_DIR"

echo "========================================"
echo "Extracting exam questions from PDFs"
echo "  Input : $EXAM_DIR"
echo "  Output: $OUT_DIR"
echo "========================================"

shopt -s nullglob
pdf_files=("$EXAM_DIR"/*.pdf)

if [ ${#pdf_files[@]} -eq 0 ]; then
    echo "No .pdf files found in $EXAM_DIR"
    exit 1
fi

total=${#pdf_files[@]}
current=1

for pdf in "${pdf_files[@]}"; do
    stem="$(basename "${pdf%.pdf}")"
    echo ""
    echo "[$current/$total] $(basename "$pdf")"
    echo "----------------------------------------"

    echo "  [pymupdf]"
    python "$SCRIPT_MU" "$pdf" "$OUT_DIR/${stem}.json" && \
        echo "  -> $OUT_DIR/${stem}.json" || \
        echo "  ERROR: pymupdf extraction failed"

    ((current++))
done

echo ""
echo "========================================"
echo "Done. Files written to $OUT_DIR"
echo "========================================"
