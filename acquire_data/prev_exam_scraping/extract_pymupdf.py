#!/usr/bin/env python3
"""
extract_pymupdf.py
Usage: python extract_pymupdf.py <pdf_path> [output_json]

Extracts exam questions from a WISEflow-exported PDF using PyMuPDF (fitz) and
saves them as structured JSON for model evaluation.

This is the PREFERRED extractor: PyMuPDF produces cleaner text than pdfplumber
for the WISEflow PDF format (no word-merging artefacts, proper ligatures, etc.).

Output structure per question:
  - question_number, points, question_type ("multiple_choice" | "open_ended")
  - question_text  : the question prompt
  - options        : {"A": "...", "B": "...", ...}  (null for open-ended)
  - correct_answer: selected option letter. WISEflow places a U+E607 glyph on
                     its own line immediately before the selected option's letter.
                     Null if absent (blank exam PDF with no answers marked).
  - model_answer   : rubric / model answer text for open-ended questions in
                     solution PDFs; null when not detected.

Requirements: pip install pymupdf
"""

import re
import json
import unicodedata
import argparse
from pathlib import Path

import fitz  # PyMuPDF

# WISEflow "selected" checkbox glyph – appears on its own line before the
# selected option's letter in PyMuPDF extractions.
_WISEFLOW_CHECK = "\ue607"

# ── Noise patterns ────────────────────────────────────────────────────────────
_NOISE = [
    re.compile(r"Forfatter\s*[-–]\s*Skriv ut", re.IGNORECASE),
    re.compile(r"wise(?:flow|ﬂow)\.net", re.IGNORECASE),  # handles ﬂ ligature
    re.compile(r"^https?://", re.IGNORECASE),              # standalone URL lines
    re.compile(r"^\d+\s+of\s+\d+\b"),                     # "1 of 7"
    re.compile(r"\d{2}/\d{2}/\d{4},?\s+\d{2}:\d{2}"),     # date/time stamp
    re.compile(r"^Seksjon\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^I have read and understood", re.IGNORECASE),
    re.compile(r"^DAT255\s+Deep Learning", re.IGNORECASE),
    re.compile(r"^Date:\s*", re.IGNORECASE),
    re.compile(r"^Time:\s*", re.IGNORECASE),
    re.compile(r"^The exam consists of", re.IGNORECASE),
    re.compile(r"^indicating that the maximum", re.IGNORECASE),
    re.compile(r"^maximum total score", re.IGNORECASE),
    re.compile(r"^You can answer in English", re.IGNORECASE),
    re.compile(r"[\xa0\s]*\d+\s*/\s*\d+\s*Word Limit", re.IGNORECASE),
    re.compile(r"[\xa0\s]*\d+\s*Word\(s\)\s*$", re.IGNORECASE),
    re.compile(r"[\xa0\s]*0Word", re.IGNORECASE),
    re.compile(r"^Comments to the questions", re.IGNORECASE),
    re.compile(r"^If you have any comments", re.IGNORECASE),
    re.compile(r"^If you have comments", re.IGNORECASE),
    re.compile(r"^Otherwise,?\s+leave this field blank", re.IGNORECASE),
    re.compile(r"^Yes\s*$"),
    re.compile(r"^No\s*$"),
    re.compile(r"^Sensor\s*[-–]\s*Skriv ut", re.IGNORECASE), # Norwegian
    re.compile(r"^Informasjonen er lest", re.IGNORECASE), # Norwegian
    re.compile(r"^\d+\s*/\s*\d+\s*Word Limit", re.IGNORECASE), # Norwegian
    re.compile(r"^europe\.wise(?:flow|ﬂow)\.net", re.IGNORECASE), # URL
]

# Rubric / model-answer text in solution PDFs tends to start with these phrases.
_RUBRIC_START = re.compile(
    r"^(Answers?\s+(should|at\s+minimum|need|include)|"
    r"Here\s+\d+\s+point|"
    r"Expected\s+answer|"
    r"We\s+expect|"
    r"For\s+full\s+marks?|"
    r"Grading:|"
    r"Rubric:)",
    re.IGNORECASE,
)

# _QUESTION_RE = re.compile(r"^Question\s+(\d+)\s*\((\d+)p\)\s*$", re.IGNORECASE)
_QUESTION_RE = re.compile(
    r"^(?:Question|Spørsmål|Oppgave)\s+(\d+)\s*\(\s*(\d+)\s*p\s*\)\s*:?\s*$",
    re.IGNORECASE
)
_OPTION_RE   = re.compile(r"^([A-E])\s*$")


def _is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    for pat in _NOISE:
        if pat.search(s):
            return True
    return False


def _normalize(text: str) -> str:
    """Normalize ligatures (ﬁ ﬂ etc.) and strip the WISEflow checkbox glyph."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_WISEFLOW_CHECK, "")
    return text.strip()


def _extract_raw_lines(pdf_path: str) -> list[str]:
    """Return all lines from the PDF, preserving content before the newline."""
    lines = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            lines.extend(text.splitlines())
    return lines


def _parse_questions(raw_lines: list[str]) -> list[dict]:
    """
    Parse question blocks from raw lines using a state machine.

    States:
      PREAMBLE       – before the first Question header
      IN_Q_TEXT      – collecting question text until first option or next Q
      IN_OPTION      – last token was an option letter; next line is option text
      IN_RUBRIC      – collecting model_answer text (open-ended solutions PDFs)

    WISEflow answer detection (PyMuPDF):
      U+E607 appears on its OWN line, immediately before the selected option's
      letter.  We set a `select_next` flag when we see it.
    """
    questions: list[dict] = []
    current: dict | None = None
    current_opt: str | None = None
    current_opt_index = 0
    select_next = False   # True when the last non-noise line was U+E607
    state = "PREAMBLE"

    for raw in raw_lines:
        stripped = raw.strip()

        # ── Bare WISEflow checkbox glyph line ────────────────────────────────
        if stripped == _WISEFLOW_CHECK or stripped == "":
            if stripped == _WISEFLOW_CHECK:
                select_next = True
            continue

        if _is_noise(raw):
            continue

        # ── Question header ──────────────────────────────────────────────────
        q_match = _QUESTION_RE.match(stripped)
        if q_match:
            if current is not None:
                questions.append(current)
            current = {
                "question_number": int(q_match.group(1)),
                "points":          int(q_match.group(2)),
                "question_text":   "",
                "question_type":   "open_ended",
                "options":         {},
                "correct_answer": [], # list of correct answer letters
                "num_answers":     0,
                "model_answer":    None,
            }
            current_opt = None
            select_next = False
            state = "IN_Q_TEXT"
            continue

        if state == "PREAMBLE" or current is None:
            select_next = False
            continue

        # ── Option letter ────────────────────────────────────────────────────
        opt_match = _OPTION_RE.match(stripped)
        if opt_match and state in ("IN_Q_TEXT", "IN_OPTION"):
            # Labeled option — existing logic unchanged
            current_opt = opt_match.group(1)
            current["question_type"] = "multiple_choice"
            if select_next:
                current["correct_answer"].append(current_opt)
                current["num_answers"] += 1
            select_next = False
            state = "IN_OPTION"
            continue
        
        # ── Unlabeled option ──────────────────────────────────────────────────
        if select_next and state == "IN_Q_TEXT":
            letter = chr(65 + current_opt_index)   # A, B, C...
            current["options"][letter] = _normalize(raw)
            current["correct_answer"].append(letter)
            current["num_answers"] += 1
            current["question_type"] = "multiple_choice"
            current_opt_index += 1
            select_next = False
            continue

        select_next = False  # reset flag for any non-letter, non-empty line

        # ── Option text ──────────────────────────────────────────────────────
        if state == "IN_OPTION" and current_opt is not None:
            current["options"][current_opt] = _normalize(raw)
            current_opt = None
            state = "IN_Q_TEXT"
            continue

        # ── Rubric / model-answer text ───────────────────────────────────────
        if state == "IN_RUBRIC":
            line = _normalize(raw)
            if current["model_answer"]:
                current["model_answer"] += " " + line
            else:
                current["model_answer"] = line
            continue

        # ── Question text ────────────────────────────────────────────────────
        if state == "IN_Q_TEXT":
            line = _normalize(raw)
            if _RUBRIC_START.match(line):
                state = "IN_RUBRIC"
                current["model_answer"] = line
                continue
            if current["question_text"]:
                current["question_text"] += " " + line
            else:
                current["question_text"] = line

    if current is not None:
        questions.append(current)

    for q in questions:
        if not q["options"]:
            q["options"] = None
        # Normalize: empty list → null, single item stays as list
        if not q["correct_answer"]:
            q["correct_answer"] = None

    return questions


def extract(pdf_path: str) -> dict:
    path = Path(pdf_path)
    raw_lines = _extract_raw_lines(pdf_path)
    questions = _parse_questions(raw_lines)
    return {
        "metadata": {
            "source_pdf":      path.name,
            "extraction_tool": "pymupdf",
            "num_questions":   len(questions),
        },
        "questions": questions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract exam questions from a WISEflow PDF using PyMuPDF (preferred).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf",    help="Path to the input PDF file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output JSON path (default: <pdf_stem>_pymupdf.json)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        parser.error(f"File not found: {pdf_path}")

    out_path = Path(args.output) if args.output else \
        pdf_path.parent / f"{pdf_path.stem}_pymupdf.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {pdf_path.name}")
    result = extract(str(pdf_path))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n   = result["metadata"]["num_questions"]
    mc  = sum(1 for q in result["questions"] if q["question_type"] == "multiple_choice")
    oe  = n - mc
    ans = sum(1 for q in result["questions"] if q.get("correct_answer"))
    ma  = sum(1 for q in result["questions"] if q.get("model_answer"))
    print(f"  {n} questions  ({mc} multiple-choice, {oe} open-ended)")
    print(f"  {ans} with detected answer,  {ma} with model answer text")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
