import re
import sys
import warnings
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import GenerationConfig, pipeline

warnings.filterwarnings(
    "ignore",
    message="You seem to be using the pipelines sequentially on GPU",
)

# ── Model config ───────────────────────────────────────────────────────────────
#
# GPU options (pick one by passing it as the second CLI argument, or change default):
#
#   "microsoft/Phi-3-mini-4k-instruct"          ~7.6 GB VRAM fp16  ← default GPU
#       Very strong instruction follower, excellent at restructuring prose.
#       Should fit on a 3080 8 GB laptop GPU.
#
#   "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"        ~4–5 GB VRAM
#       4-bit quantized 7B — best quality/VRAM ratio. Requires:
#           pip install auto-gptq optimum
#
#   "Qwen/Qwen2.5-7B-Instruct" (bnb 4-bit)      ~4–5 GB VRAM
#       Load with load_in_4bit=True. Requires:
#           pip install bitsandbytes
#       Pass "Qwen/Qwen2.5-7B-Instruct:bnb4" as model_id to use this path.
#
# CPU fallback:
#   "Qwen/Qwen2.5-1.5B-Instruct"  — slow but runnable

CPU_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GPU_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# ── Chunk limits (words) ───────────────────────────────────────────────────────
CPU_MAX_CHUNK_WORDS = 250
GPU_MAX_CHUNK_WORDS = 1000   # tighter than refine — summarization needs focus

# ── Output compression target ──────────────────────────────────────────────────
# max_new_tokens = input_words * this factor.  Lower = more aggressive compression.
# 0.6 targets ~40% reduction on top of whatever filler removal already did.
CPU_COMPRESSION = 0.6
GPU_COMPRESSION = 0.6

# ── Prompts ────────────────────────────────────────────────────────────────────

CPU_SYSTEM = (
    "Summarize this lecture excerpt into academic notes. "
    "Remove all greetings, admin, and conversational content. "
    "Keep only technical facts and concepts. "
    "Write in third-person academic prose. "
    "Output only the summary."
)
CPU_USER_TEMPLATE = "INPUT:\n{text}\n\nSUMMARY:"

# ---------------------------------------------------------------------------
GPU_SYSTEM = """\
You convert spoken lecture transcripts into concise academic notes.

REMOVE without exception:
- Greetings, sign-offs, sound checks ("Good morning", "Hello", "okay let's start")
- Administrative content (what was covered last week, lab setup, Zoom issues)
- References to the audience ("you should", "we will", "as you remember", "I will show you")
- Phrases that describe what the lecturer is ABOUT to say instead of saying it
  ("So now we will look at X" → just state X directly)
- Filler transitions ("So...", "And then...", "Okay so...", "Right so...")
- Repetition of the same point within one paragraph

KEEP:
- Every technical definition, concept, formula, and named method
- Every concrete example that illustrates a concept
- The logical structure and progression of ideas

REWRITE:
- First/second person → third person academic prose
  ("we use layers" → "layers are used", "you can imagine" → "consider")
- Run-on unpunctuated speech → properly punctuated sentences
- Target roughly 40–50% of the original word count

Output only the condensed academic text. No bullet points, no labels, no commentary."""

GPU_USER_TEMPLATE = """\
Example — match this level of condensing:

BEFORE:
"Good morning, it's time and we have a lot of stuff today, so let's get started \
once I of course do the quick Zoom sound check. Hello. Okay, so no response, \
let's, ah, there we go, yes, thank you. So let's do deep learning this week, \
finally. So we spent a week looking at some examples and you're supposed to set \
up some kind of working setup for yourself last week. So hopefully this works \
and then finally this week we get to test it. So we discussed a lot on deep \
learning and we looked at some examples and now we want to actually see how this \
works and start building our own models and the topic for now the next three \
weeks I think will be deep learning on image inputs. So we want to do something \
like this, take an image and then maybe find, well, a car, you could put a box \
around to mark the location if you want or you can have recognition of people \
and count them if you like."

AFTER:
"The next three weeks focus on deep learning applied to image inputs. Computer \
vision tasks include image classification, object detection with bounding boxes, \
and person counting."

Now condense the following transcript to the same standard:
{text}"""


# ── Hallucination detection ────────────────────────────────────────────────────
_HALLUCINATION_RE = re.compile(
    r"^sure,?\s+here"
    r"|^here'?s?\s+the\s+(summ|clean|condensed)"
    r"|^i'?ll\s+(summ|clean|condense)"
    r"|^(summary|condensed version|academic notes):"
    r"|^\*\*"
    r"|^---"
    r"|^please\s+(go|provide|share)"
    r"|^note that"
    r"|^output:",
    re.IGNORECASE | re.MULTILINE,
)
_WRAPPER_RE     = re.compile(
    r'^(\*{1,3}|---|"|\bsure\b|\bhere\b|\bi\'ll\b|\bsummary\b|\boutput\b)',
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


# Matches timestamps like "(10:12)", "(1:02:45)", "[ 10:12 ]" at end of heading
_TIMESTAMP_RE = re.compile(r'\s*[\(\[]\s*\d{1,2}:\d{2}(:\d{2})?\s*[\)\]]\s*$')


def _strip_timestamp(header: str) -> str:
    """Remove trailing timestamp from a markdown heading, e.g. '## Intro (10:12)' → '## Intro'."""
    return _TIMESTAMP_RE.sub("", header).rstrip()


def rebuild_markdown(sections: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for header, body in sections:
        clean_header = _strip_timestamp(header) if header else header
        is_title     = clean_header.startswith("# ")  and not clean_header.startswith("## ")
        is_section   = bool(clean_header) and not is_title
        stripped     = body.strip()
        if is_title:
            lines.append(clean_header)
        elif is_section:
            lines.append("")
            lines.append(clean_header)
        elif clean_header:
            lines.append("")
            lines.append(clean_header)
        if stripped:
            lines.append("")
            lines.append(stripped)
    return "\n".join(lines) + "\n"


# ── Section chunking ───────────────────────────────────────────────────────────

def _split_into_chunks(text: str, max_words: int) -> list[str]:
    """
    Split on sentence boundaries. Falls back to clause splitting for
    unpunctuated spoken text (long runs with no . ? !).
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) == 1:
        # No punctuation — split at spoken clause joiners
        sentences = re.split(
            r'\s+(?=\b(?:and|but|so|then|also|okay|now|which|where|when)\b)',
            text.strip(),
        )

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        w = len(sentence.split())
        if current_words + w > max_words and current:
            chunks.append(" ".join(current))
            current, current_words = [], 0
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

    # Special case: "model_id:bnb4" triggers 4-bit bitsandbytes quantization
    if model_id.endswith(":bnb4"):
        real_id = model_id[:-5]
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            print(f"--- Loading {real_id} (4-bit bnb) on GPU ---")
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(real_id)
            model     = AutoModelForCausalLM.from_pretrained(
                real_id, quantization_config=bnb_cfg, device_map="auto"
            )
            return pipeline("text-generation", model=model, tokenizer=tokenizer)
        except ImportError:
            print("    [WARN] bitsandbytes not installed — pip install bitsandbytes")
            print("    [WARN] falling back to standard fp16 loading")
            model_id = real_id

    device = 0 if use_gpu else -1
    dtype  = torch.float16 if use_gpu else torch.float32
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
    """
    Returns True only when the output is very likely cut off mid-word or mid-clause.
    We accept any punctuation or closing bracket as a valid ending — Phi-3 often
    ends with ')', ']', or a closing quote variant.
    """
    stripped = text.rstrip()
    if not stripped:
        return False
    last = stripped[-1]
    # Ends mid-word: last char is alphanumeric (no punctuation at all)
    # Also flag common mid-clause connectors as the last word
    if last.isalnum():
        last_word = stripped.split()[-1].lower().rstrip(",'")
        mid_clause = {"and", "or", "but", "the", "a", "an", "of", "in", "to",
                      "for", "with", "that", "which", "where", "when", "so"}
        return last_word in mid_clause or len(stripped) < 30
    return False  # ends with any punctuation → accept it


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
        # A truncated summary is still better than the full raw original.
        # Append an ellipsis so it's clear the thought was cut, but keep it.
        print("    [WARN] output appears truncated — keeping summary with ellipsis")
        return raw.rstrip() + "…"
    return raw


# ── Single-chunk summarization ─────────────────────────────────────────────────

def _summarize_chunk(
    pipe, chunk: str, system_prompt: str, user_template: str, compression: float
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_template.format(text=chunk.strip())},
    ]
    # Target ~60% of input length, but never below 150 tokens.
    # The old floor of 64 was too low for sections with 100-200 words.
    max_new_tokens = max(150, int(len(chunk.split()) * compression))
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    result = pipe(messages, generation_config=gen_cfg)
    raw    = result[0]["generated_text"][-1]["content"].strip()
    return sanitize_output(raw, original=chunk)


# ── Section-level summarization ────────────────────────────────────────────────

def summarize_section(
    pipe,
    text: str,
    system_prompt: str,
    user_template: str,
    max_chunk_words: int,
    compression: float,
) -> str:
    if not text.strip():
        return text
    if len(text.split()) <= max_chunk_words:
        return _summarize_chunk(pipe, text, system_prompt, user_template, compression)

    chunks  = _split_into_chunks(text, max_chunk_words)
    refined = [
        _summarize_chunk(pipe, c, system_prompt, user_template, compression)
        for c in chunks
    ]
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
    if folder_name.endswith("_refined"):
        new_folder_name = folder_name[:-8] + "_summary"
    else:
        new_folder_name = folder_name + "_summary"

    # Rebuild the parent path with the modified folder name
    new_parent = parent.parent / new_folder_name

    # Keep the original filename unchanged
    return new_parent / input_path.name


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python lecture_transcript_summarize.py <input.md> [model_id]")
        print()
        print("  model_id examples:")
        print(f"    (default GPU) {GPU_MODEL}")
        print(f"    (default CPU) {CPU_MODEL}")
        print("    Qwen/Qwen2.5-7B-Instruct:bnb4   ← 4-bit quant, needs bitsandbytes")
        print("    Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4  ← GPTQ quant, needs auto-gptq")
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
    compression     = GPU_COMPRESSION   if use_gpu else CPU_COMPRESSION

    output_md.parent.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_md}")
    print(f"Output: {output_md}")
    print(f"Mode:   {'GPU' if use_gpu else 'CPU'}  |  model: {model_id}  |  max chunk: {max_chunk_words} words  |  compression: {compression}")

    content  = input_md.read_text(encoding="utf-8")
    sections = parse_sections(content)
    pipe     = load_model(model_id)

    print("--- Summarizing sections ---")
    summarized_sections: list[tuple[str, str]] = []

    for header, body in tqdm(sections, desc="Sections", unit="section", ncols=100):
        if not body.strip():
            summarized_sections.append((header, body))
        else:
            summarized_body = summarize_section(
                pipe, body, system_prompt, user_template, max_chunk_words, compression
            )
            summarized_sections.append((header, summarized_body))

    output_md.write_text(rebuild_markdown(summarized_sections), encoding="utf-8")
    print(f"\n--- Done! Saved to {output_md} ---")


if __name__ == "__main__":
    main()