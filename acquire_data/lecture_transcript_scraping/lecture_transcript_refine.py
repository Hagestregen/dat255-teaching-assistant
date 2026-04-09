import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline

# ── Model config ───────────────────────────────────────────────────────────────
# Tier 1 — CPU-friendly (current)
#   "Qwen/Qwen2.5-0.5B-Instruct"     ~1 GB  — often hallucinates on messy input
#   "Qwen/Qwen2.5-1.5B-Instruct"     ~3 GB  — much more reliable, still CPU-usable
#
# Tier 2 — GPU recommended
#   "microsoft/Phi-3-mini-4k-instruct"   ~7 GB  — excellent instruction following
#   "Qwen/Qwen2.5-7B-Instruct"           ~15 GB — high quality
#
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Prompt ─────────────────────────────────────────────────────────────────────
# Rules for writing prompts that work on small (0.5B–1.5B) models:
#  - One job per prompt, stated in the first sentence
#  - Show an example (few-shot) — far more reliable than rule lists
#  - End with a clear output marker so the model knows where to start writing

SYSTEM_PROMPT = """\
You remove filler words from spoken transcripts. Output only the cleaned transcript text, nothing else."""

# Few-shot example embedded in the user turn (more reliable than system rules on tiny models)
FEW_SHOT_PREFIX = """\
Example:
INPUT: "Um, so yeah, the, uh, gradient descent algorithm, you know, it kind of updates the weights, right, based on the loss."
OUTPUT: "The gradient descent algorithm updates the weights based on the loss."

Now clean this transcript. Output only the cleaned text, no explanations:
"""

# ── Patterns that indicate the model went off-rails ───────────────────────────
# If the model's output contains any of these, it has hallucinated meta-commentary.
HALLUCINATION_MARKERS = [
    r"^sure,?\s+here",
    r"^here'?s?\s+the\s+clean",
    r"^i'?ll\s+clean",
    r"^cleaned\s+text",
    r"^i'?m\s+ready",
    r"^\*\*cleaned",
    r"^---",
    r"^1\.\s+remove",          # model explaining the rules back
    r"^please\s+(go|provide|share)",
    r"^note that",
    r"^output:",
]
_HALLUCINATION_RE = re.compile(
    "|".join(HALLUCINATION_MARKERS), re.IGNORECASE | re.MULTILINE
)

# Lines that are clearly model meta-commentary (not transcript content)
_META_LINE_RE = re.compile(
    r"^(sure|here|i'll|cleaned|ready|note|please|output:|---|\*\*|\d+\.)",
    re.IGNORECASE,
)


# ── Markdown parsing ───────────────────────────────────────────────────────────

def parse_sections(content: str) -> list[tuple[str, str]]:
    pattern = r'^(#{1,6} .+)$'
    parts = re.split(pattern, content, flags=re.MULTILINE)
    sections: list[tuple[str, str]] = []
    if parts[0].strip():
        sections.append(("", parts[0]))
    i = 1
    while i < len(parts):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((header, body))
        i += 2
    return sections


def rebuild_markdown(sections: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    prev_was_title = False

    for header, body in sections:
        is_title   = header.startswith("# ") and not header.startswith("## ")
        is_section = len(header) > 0 and not is_title
        stripped_body = body.strip()

        if is_title:
            lines.append(header)
            prev_was_title = True
        elif is_section:
            lines.append("")          # blank line before every non-title heading
            lines.append(header)
            prev_was_title = False
        elif header:
            lines.append("")
            lines.append(header)
            prev_was_title = False
        else:
            prev_was_title = False

        if stripped_body:
            lines.append("")
            lines.append(stripped_body)

    return "\n".join(lines) + "\n"


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_id: str):
    device = 0 if torch.cuda.is_available() else -1
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"--- Loading {model_id} on {'GPU' if device == 0 else 'CPU'} ---")
    print("    (downloaded once to ~/.cache/huggingface)")
    return pipeline(
        "text-generation",
        model=model_id,
        device=device,
        torch_dtype=dtype,
    )


# ── Output sanitization ────────────────────────────────────────────────────────

def _looks_like_hallucination(text: str) -> bool:
    """Returns True if the model output is meta-commentary rather than cleaned text."""
    first_500 = text[:500]
    return bool(_HALLUCINATION_RE.search(first_500))


def _strip_meta_preamble(text: str) -> str:
    """
    If the model prefixed its answer with an explanation or header block,
    try to extract just the actual transcript content.

    Strategy: find the first line that looks like real prose (starts with
    a capital or lowercase letter, is reasonably long, and doesn't match
    known meta patterns).
    """
    lines = text.splitlines()
    real_lines = []
    found_content = False

    # Common wrapper patterns the model wraps around the real answer
    wrapper_re = re.compile(
        r'^(\*{1,3}|---|"|\bsure\b|\bhere\b|\bi\'ll\b|\bcleaned\b|\boutput\b)',
        re.IGNORECASE
    )
    # A line that looks like a quoted block start/end: just a quote mark or dashes
    quote_fence_re = re.compile(r'^["\'`]{1,3}$|^-{3,}$')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if found_content:
                real_lines.append("")
            continue
        if wrapper_re.match(stripped) or quote_fence_re.match(stripped):
            # Could be the wrapper around the real answer — skip
            continue
        # Heuristic: real transcript lines are usually > 20 chars
        if not found_content and len(stripped) < 15:
            continue
        found_content = True
        real_lines.append(line)

    # Strip trailing empty lines
    while real_lines and not real_lines[-1].strip():
        real_lines.pop()

    return "\n".join(real_lines)


def sanitize_output(raw_output: str, original_text: str) -> str:
    """
    Main output guard. Returns cleaned text, or falls back to original
    if the model clearly went off-rails and we can't recover anything useful.
    """
    if not raw_output.strip():
        return original_text

    if _looks_like_hallucination(raw_output):
        recovered = _strip_meta_preamble(raw_output)
        if len(recovered.strip()) > 30:   # recovered something real
            return recovered
        else:
            # Total failure — return the original with a warning comment
            print("    [WARN] model hallucinated, keeping original text")
            return original_text

    return raw_output


# ── Refinement ─────────────────────────────────────────────────────────────────

def refine_text(pipe, text: str) -> str:
    if not text.strip():
        return ""

    user_content = FEW_SHOT_PREFIX + text.strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # Give the model a little headroom — cleaning never makes text longer
    max_tokens = max(64, int(len(text.split()) * 1.05))

    result = pipe(
        messages,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    raw = result[0]["generated_text"][-1]["content"].strip()
    return sanitize_output(raw, original_text=text)


# ── Path helpers ───────────────────────────────────────────────────────────────

def derive_output_path(input_path: Path) -> Path:
    parts = list(input_path.parts)
    new_parts = [p[:-4] if p.endswith("_raw") else p for p in parts]
    return Path(*new_parts)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python lecture_transcript_refine.py <input.md> [model_id]")
        print(f"       model_id defaults to {DEFAULT_MODEL}")
        sys.exit(1)

    input_md  = Path(sys.argv[1])
    model_id  = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL
    output_md = derive_output_path(input_md)

    if not input_md.exists():
        print(f"Error: {input_md} not found.")
        sys.exit(1)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_md}")
    print(f"Output: {output_md}")

    content  = input_md.read_text(encoding="utf-8")
    sections = parse_sections(content)
    pipe     = load_model(model_id)

    print("--- Refining sections ---")
    refined_sections: list[tuple[str, str]] = []

    for header, body in tqdm(sections, desc="Sections", unit="section", ncols=100):
        is_title = header.startswith("# ") and not header.startswith("## ")
        if is_title or not body.strip():
            refined_sections.append((header, body))
        else:
            refined_body = refine_text(pipe, body)
            refined_sections.append((header, refined_body))

    output_md.write_text(rebuild_markdown(refined_sections), encoding="utf-8")
    print(f"\n--- Done! Saved to {output_md} ---")


if __name__ == "__main__":
    main()