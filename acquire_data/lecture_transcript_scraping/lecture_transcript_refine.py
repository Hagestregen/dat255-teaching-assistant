import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import GenerationConfig, pipeline

# ── Model config ───────────────────────────────────────────────────────────────
#
# CPU — small, reliable, fast enough
#   "Qwen/Qwen2.5-1.5B-Instruct"     ~3 GB RAM   ← default CPU choice
#   "Qwen/Qwen2.5-0.5B-Instruct"     ~1 GB RAM   — often hallucinates on messy input
#
# GPU (laptop 3080, ~8 GB VRAM)
#   "Qwen/Qwen2.5-3B-Instruct"       ~6 GB VRAM  ← default GPU choice, handles long context
#   "microsoft/Phi-3-mini-4k-instruct"  ~7 GB VRAM  — excellent quality, tight on 8 GB
#   "Qwen/Qwen2.5-7B-Instruct"       ~15 GB VRAM — needs 16 GB

CPU_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GPU_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ── Chunk / token limits ───────────────────────────────────────────────────────
# Sections larger than this (in words) are split into smaller chunks before
# sending to the model.  Keeping inputs short is especially important on CPU
# where a 600-word section already takes ~15–20 s on a 1.5B model.
#
# Rule of thumb: 1 word ≈ 1.3 tokens for English prose.
CPU_MAX_CHUNK_WORDS = 500    # ~650 tokens  →  reasonable CPU speed
GPU_MAX_CHUNK_WORDS = 2000   # ~2600 tokens →  comfortably within 3B context window

# ── Prompts ────────────────────────────────────────────────────────────────────
# Small (CPU) models follow short, imperative prompts best.
# Larger (GPU) models benefit from a concrete few-shot example.

CPU_SYSTEM = (
    "Remove filler words from this spoken transcript. "
    "Output only the cleaned text, nothing else."
)

# The input/output marker keeps small models from re-explaining the task.
CPU_USER_TEMPLATE = "INPUT:\n{text}\n\nOUTPUT:"

GPU_SYSTEM = (
    "You remove filler words from spoken lecture transcripts.\n"
    "Output only the cleaned transcript text — no explanations, no labels, nothing else."
)

GPU_USER_TEMPLATE = """\
Example
INPUT: "Um, so yeah, the gradient descent algorithm, you know, it kind of updates the weights, right, based on the loss."
OUTPUT: "The gradient descent algorithm updates the weights based on the loss."

Now clean the following transcript. Output only the cleaned text:
{text}"""

# ── Hallucination detection ────────────────────────────────────────────────────
_HALLUCINATION_RE = re.compile(
    r"^sure,?\s+here"
    r"|^here'?s?\s+the\s+clean"
    r"|^i'?ll\s+clean"
    r"|^cleaned\s+text"
    r"|^i'?m\s+ready"
    r"|^\*\*cleaned"
    r"|^---"
    r"|^1\.\s+remove"
    r"|^please\s+(go|provide|share)"
    r"|^note that"
    r"|^output:",
    re.IGNORECASE | re.MULTILINE,
)

_WRAPPER_RE = re.compile(
    r'^(\*{1,3}|---|"|\bsure\b|\bhere\b|\bi\'ll\b|\bcleaned\b|\boutput\b)',
    re.IGNORECASE,
)
_QUOTE_FENCE_RE = re.compile(r'^["\'`]{1,3}$|^-{3,}$')


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
    """
    Split *text* into chunks of at most *max_words* words,
    preferring sentence boundaries ('. ', '.\n').
    """
    # Tokenise on sentence endings first, then fall back to word-count hard split.
    sentence_end = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_end.split(text.strip())

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        w = len(sentence.split())
        if current_words + w > max_words and current:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        # A single sentence that is itself over the limit gets hard-split by words.
        if w > max_words:
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


def sanitize_output(raw: str, original: str) -> str:
    if not raw.strip():
        return original
    if _HALLUCINATION_RE.search(raw[:500]):
        recovered = _strip_meta_preamble(raw)
        if len(recovered.strip()) > 30:
            return recovered
        print("    [WARN] model hallucinated — keeping original text for this chunk")
        return original
    return raw


# ── Single-chunk refinement ────────────────────────────────────────────────────

def _refine_chunk(pipe, chunk: str, system_prompt: str, user_template: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_template.format(text=chunk.strip())},
    ]
    max_new_tokens = max(64, int(len(chunk.split()) * 1.1))

    # Fix: use a GenerationConfig object so we don't mix config + explicit kwargs
    # (avoids the "Passing generation_config together with generation-related
    #  arguments is deprecated" warning).
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
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

    word_count = len(text.split())
    if word_count <= max_chunk_words:
        return _refine_chunk(pipe, text, system_prompt, user_template)

    # Section is too large — split, refine each chunk, rejoin.
    chunks  = _split_into_chunks(text, max_chunk_words)
    refined = [_refine_chunk(pipe, c, system_prompt, user_template) for c in chunks]
    return " ".join(refined)


# ── Path helpers ───────────────────────────────────────────────────────────────

def derive_output_path(input_path: Path) -> Path:
    parts     = list(input_path.parts)
    new_parts = [p[:-4] if p.endswith("_raw") else p for p in parts]
    return Path(*new_parts)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python lecture_transcript_refine.py <input.md> [model_id]")
        sys.exit(1)

    input_md = Path(sys.argv[1])
    if not input_md.exists():
        print(f"Error: {input_md} not found.")
        sys.exit(1)

    use_gpu   = torch.cuda.is_available()
    default_model = GPU_MODEL if use_gpu else CPU_MODEL

    model_id       = sys.argv[2] if len(sys.argv) >= 3 else default_model
    output_md      = derive_output_path(input_md)
    system_prompt  = GPU_SYSTEM  if use_gpu else CPU_SYSTEM
    user_template  = GPU_USER_TEMPLATE if use_gpu else CPU_USER_TEMPLATE
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