import re
import sys
import warnings
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import GenerationConfig, pipeline

# Suppress the "pipelines sequentially on GPU" info message — we are doing this
# intentionally (one section at a time) and the Dataset API adds complexity we don't need.
warnings.filterwarnings(
    "ignore",
    message="You seem to be using the pipelines sequentially on GPU",
)

# ── Model config ───────────────────────────────────────────────────────────────
CPU_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GPU_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ── Chunk limits (words) ───────────────────────────────────────────────────────
CPU_MAX_CHUNK_WORDS = 300   # small model quality drops fast with long input
GPU_MAX_CHUNK_WORDS = 1500

# ── Prompts ────────────────────────────────────────────────────────────────────
#
# Key insight: small (CPU) models follow short, imperative prompts best.
# Larger (GPU) models benefit from an explicit filler list + a dramatic example.
#
# The example below is intentionally "bad input → very clean output" to push
# the model toward being aggressive rather than timid.

CPU_SYSTEM = (
    "Clean this spoken transcript. "
    "Remove filler words (um, uh, so, you know, right, kind of, basically, well, okay, like, I mean). "
    "Fix grammar after removal. Output only the cleaned text."
)
CPU_USER_TEMPLATE = "INPUT:\n{text}\n\nOUTPUT:"

# ---------------------------------------------------------------------------
GPU_SYSTEM = """\
You clean spoken lecture transcripts into readable prose.

REMOVE all of the following:
- Filler words: um, uh, so (as filler), you know, right (as filler), kind of,
  sort of, basically, well (as filler), okay (as filler), I mean, actually (as
  filler at start of clause), like (as filler)
- Sentence-opening clutter: "So...", "And so...", "Okay so...", "And then..."
- False starts and self-corrections: "we, at least those of you who are..."
- Obvious repetitions of the same idea within one sentence

KEEP every technical fact, example, analogy, and explanation.
Fix grammar and sentence flow after removing clutter.
Output only the cleaned text — no labels, no commentary."""

GPU_USER_TEMPLATE = """\
Example — this is the level of cleaning expected:

BEFORE:
"Yes, it's time, so let's get started. So many of the, at least those of you \
who are master students should be on this research seminar. So that's why we \
are a bit fewer people, but okay. So still a very important topic for today. \
So we're now starting our natural language processing."

AFTER:
"Let's get started. Master students should be on the research seminar, which \
is why we have fewer people today. It's still a very important topic. We're \
now starting natural language processing."

Now clean the following transcript to the same standard:
{text}"""


# ── Hallucination detection ────────────────────────────────────────────────────
_HALLUCINATION_RE = re.compile(
    r"^sure,?\s+here"
    r"|^here'?s?\s+the\s+clean"
    r"|^i'?ll\s+clean"
    r"|^cleaned\s+(text|transcript|version)"
    r"|^i'?m\s+ready"
    r"|^\*\*cleaned"
    r"|^---"
    r"|^1\.\s+remove"
    r"|^please\s+(go|provide|share)"
    r"|^note that"
    r"|^output:",
    re.IGNORECASE | re.MULTILINE,
)
_WRAPPER_RE     = re.compile(
    r'^(\*{1,3}|---|"|\bsure\b|\bhere\b|\bi\'ll\b|\bcleaned\b|\boutput\b)',
    re.IGNORECASE,
)
_QUOTE_FENCE_RE = re.compile(r'^["\'`]{1,3}$|^-{3,}$')


# ── Markdown parsing ───────────────────────────────────────────────────────────

def parse_sections(content: str) -> list[tuple[str, str]]:
    parts = re.split(r'^(#{1,6} .+)$', content, flags=re.MULTILINE)
    sections: list[tuple[str, str]] = []
    if parts[0].strip():
        sections.append(("", parts[0]))
    i = 1
    while i < len(parts):
        header = parts[i].strip()
        body   = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((header, body))
        i += 2
    return sections


def rebuild_markdown(sections: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for header, body in sections:
        is_title   = header.startswith("# ")  and not header.startswith("## ")
        is_section = bool(header) and not is_title
        stripped   = body.strip()
        if is_title:
            lines.append(header)
        elif is_section:
            lines.append("")
            lines.append(header)
        elif header:
            lines.append("")
            lines.append(header)
        if stripped:
            lines.append("")
            lines.append(stripped)
    return "\n".join(lines) + "\n"


# ── Section chunking ───────────────────────────────────────────────────────────

def _split_into_chunks(text: str, max_words: int) -> list[str]:
    """Split on sentence boundaries; hard-split any sentence that is itself too long."""
    sentences   = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        w = len(sentence.split())
        if current_words + w > max_words and current:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        if w > max_words:                       # single giant sentence — word-split it
            words = sentence.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
        else:
            current.append(sentence)
            current_words += w

    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_id: str):
    use_gpu = torch.cuda.is_available()
    device  = 0 if use_gpu else -1
    dtype   = torch.float16 if use_gpu else torch.float32
    print(f"--- Loading {model_id} on {'GPU' if use_gpu else 'CPU'} ---")
    print("    (downloaded once to ~/.cache/huggingface)")
    return pipeline("text-generation", model=model_id, device=device, dtype=dtype)


# ── Output sanitization ────────────────────────────────────────────────────────

def _strip_meta_preamble(text: str) -> str:
    lines, real_lines, found = text.splitlines(), [], False
    for line in lines:
        s = line.strip()
        if not s:
            if found:
                real_lines.append("")
            continue
        if _WRAPPER_RE.match(s) or _QUOTE_FENCE_RE.match(s):
            continue
        if not found and len(s) < 15:
            continue
        found = True
        real_lines.append(line)
    while real_lines and not real_lines[-1].strip():
        real_lines.pop()
    return "\n".join(real_lines)


def _ends_mid_sentence(text: str) -> bool:
    """Heuristic: output probably got cut off if it doesn't end with punctuation."""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] not in ".!?\"'"


def sanitize_output(raw: str, original: str) -> str:
    if not raw.strip():
        return original
    if _HALLUCINATION_RE.search(raw[:500]):
        recovered = _strip_meta_preamble(raw)
        if len(recovered.strip()) > 30:
            return recovered
        print("    [WARN] hallucination detected — keeping original text for this chunk")
        return original
    if _ends_mid_sentence(raw):
        print("    [WARN] output appears truncated — keeping original text for this chunk")
        return original
    return raw


# ── Single-chunk refinement ────────────────────────────────────────────────────

def _refine_chunk(pipe, chunk: str, system_prompt: str, user_template: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_template.format(text=chunk.strip())},
    ]

    # Give generous headroom — output is always shorter than input, so this is
    # cheap on GPU and avoids mid-sentence truncation.
    # Explicitly resetting temperature/top_p/top_k silences the Qwen warning
    # about "generation flags not valid".
    max_new_tokens = max(256, int(len(chunk.split()) * 1.5))
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,   # ignored when do_sample=False; set explicitly to
        top_p=1.0,         # prevent "generation flags not valid" warning
    )

    result = pipe(messages, generation_config=gen_cfg)
    raw    = result[0]["generated_text"][-1]["content"].strip()
    return sanitize_output(raw, original=chunk)


# ── Section-level refinement (with auto-chunking) ─────────────────────────────

def refine_section(
    pipe,
    text: str,
    system_prompt: str,
    user_template: str,
    max_chunk_words: int,
) -> str:
    if not text.strip():
        return text
    if len(text.split()) <= max_chunk_words:
        return _refine_chunk(pipe, text, system_prompt, user_template)

    chunks  = _split_into_chunks(text, max_chunk_words)
    refined = [_refine_chunk(pipe, c, system_prompt, user_template) for c in chunks]
    return " ".join(refined)


# ── Path helpers ───────────────────────────────────────────────────────────────

def derive_output_path(input_path: Path) -> Path:
    parent = input_path.parent
    folder_name = parent.name

    # If there is no actual folder name (e.g. file in root or current dir),
    # leave the path unchanged since there is no folder to alter
    if not folder_name or folder_name == ".":
        return input_path

    # Apply the requested rule to the folder name only
    if folder_name.endswith("_raw"):
        new_folder_name = folder_name[:-4] + "_refined"
    else:
        new_folder_name = folder_name + "_refined"

    # Rebuild the parent path with the modified folder name
    new_parent = parent.parent / new_folder_name

    # Keep the original filename unchanged
    return new_parent / input_path.name


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python lecture_transcript_refine.py <input.md> [model_id]")
        sys.exit(1)

    input_md = Path(sys.argv[1])
    if not input_md.exists():
        print(f"Error: {input_md} not found.")
        sys.exit(1)

    use_gpu         = torch.cuda.is_available()
    default_model   = GPU_MODEL if use_gpu else CPU_MODEL
    model_id        = sys.argv[2] if len(sys.argv) >= 3 else default_model
    output_md       = derive_output_path(input_md)
    system_prompt   = GPU_SYSTEM        if use_gpu else CPU_SYSTEM
    user_template   = GPU_USER_TEMPLATE if use_gpu else CPU_USER_TEMPLATE
    max_chunk_words = GPU_MAX_CHUNK_WORDS if use_gpu else CPU_MAX_CHUNK_WORDS

    output_md.parent.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_md}")
    print(f"Output: {output_md}")
    print(f"Mode:   {'GPU' if use_gpu else 'CPU'}  |  model: {model_id}  |  max chunk: {max_chunk_words} words")

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
            refined_body = refine_section(
                pipe, body, system_prompt, user_template, max_chunk_words
            )
            refined_sections.append((header, refined_body))

    output_md.write_text(rebuild_markdown(refined_sections), encoding="utf-8")
    print(f"\n--- Done! Saved to {output_md} ---")


if __name__ == "__main__":
    main()