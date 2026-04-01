import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline

# --- CONFIGURATION ---
# Model is downloaded once to HuggingFace's cache (~/.cache/huggingface) and reused on subsequent runs.
# Swap for a larger model if quality isn't sufficient:
#   "Qwen/Qwen2.5-1.5B-Instruct"  (~3 GB, noticeably better)
#   "microsoft/Phi-3-mini-4k-instruct"  (~7 GB, high quality)
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1 GB, fast on CPU

SYSTEM_PROMPT = """\
You are cleaning up a university lecture transcript.
Your rules:
- REMOVE verbal fillers: "um", "ah", "uh", "okay so", "so yeah", "right", "you know", \
audio check lines ("hello hello", "can you hear me"), and false starts.
- KEEP every technical term, definition, example, formula, and named concept exactly as written.
- Do NOT summarize, shorten, or paraphrase the actual content.
- Do NOT add any headings, bullet points, or formatting — output plain prose only.
- Output only the cleaned text. No explanations, no preamble."""


# --- MARKDOWN PARSING ---

def parse_sections(content: str) -> list[tuple[str, str]]:
    """
    Splits markdown into a list of (header_line, body_text) pairs.
    The leading # title (level-1) is returned as ("", body) so it is never refined.
    All ## section headers are included as their own header string.
    """
    # Split on any ATX heading line
    pattern = r'^(#{1,6} .+)$'
    parts = re.split(pattern, content, flags=re.MULTILINE)

    sections: list[tuple[str, str]] = []
    # parts alternates: [pre-header text, header, body, header, body, ...]
    if parts[0].strip():
        sections.append(("", parts[0]))  # content before any heading (rare)

    i = 1
    while i < len(parts):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((header, body))
        i += 2

    return sections


def rebuild_markdown(sections: list[tuple[str, str, str]]) -> str:
    """
    Reassembles (header, refined_body) pairs into a clean markdown string.
    Formatting rules:
      - # title: no blank line after (first ## handles the gap)
      - ## headers: blank line before (except right after # title) and after
      - Adjacent headers with no body between them: only one blank line between them
    """
    lines: list[str] = []
    prev_was_header = False
    prev_was_title = False

    for header, body in sections:
        is_title = header.startswith("# ") and not header.startswith("## ")
        is_section = header.startswith("## ")
        stripped_body = body.strip()

        if is_title:
            lines.append(header)
            prev_was_title = True
            prev_was_header = True
        elif is_section:
            if not prev_was_title:
                lines.append("")   # blank line before ## (unless right after title)
            lines.append(header)
            prev_was_title = False
            prev_was_header = True
        elif header:
            # Deeper heading (###, ####, …) — treat same as ##
            lines.append("")
            lines.append(header)
            prev_was_title = False
            prev_was_header = True
        else:
            prev_was_header = False
            prev_was_title = False

        if stripped_body:
            lines.append("")       # blank line after header before body
            lines.append(stripped_body)
            prev_was_header = False

    return "\n".join(lines) + "\n"


# --- LLM REFINEMENT ---

def load_model(model_id: str):
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"--- Loading {model_id} on {'GPU' if device == 0 else 'CPU'} ---")
    print("    (downloads once to ~/.cache/huggingface, then cached)")
    return pipeline(
        "text-generation",
        model=model_id,
        device=device,
        torch_dtype=dtype,
    )


def refine_text(pipe, text: str) -> str:
    """Sends a block of transcript text through the instruct model for cleanup."""
    if not text.strip():
        return ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text.strip()},
    ]

    # Allow slightly fewer tokens than the input — cleaning shortens text, never lengthens it
    max_tokens = max(64, int(len(text.split()) * 1.1))

    result = pipe(
        messages,
        max_new_tokens=max_tokens,
        do_sample=False,          # deterministic / no hallucination drift
        temperature=None,
        top_p=None,
    )
    return result[0]["generated_text"][-1]["content"].strip()


# --- PATH HELPERS ---

def derive_output_path(input_path: Path) -> Path:
    """
    Replaces the *_raw directory segment with its non-raw counterpart.
    e.g. data/lecture_transcript_raw/lec1.md -> data/lecture_transcript/lec1.md
    """
    parts = list(input_path.parts)
    new_parts = [p[:-4] if p.endswith("_raw") else p for p in parts]  # strip trailing _raw
    return Path(*new_parts)


# --- MAIN ---

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

    pipe = load_model(model_id)

    print("--- Refining sections ---")
    refined_sections: list[tuple[str, str]] = []

    # Sections that should NOT be refined (headings only, title line, empty bodies)
    for header, body in tqdm(sections, desc="Sections", unit="section", ncols=100):
        is_title = header.startswith("# ") and not header.startswith("## ")

        if is_title or not body.strip():
            # Never touch the title or bodyless headers
            refined_sections.append((header, body))
        else:
            refined_body = refine_text(pipe, body)
            refined_sections.append((header, refined_body))

    output_md.write_text(rebuild_markdown(refined_sections), encoding="utf-8")
    print(f"\n--- Done! Saved to {output_md} ---")


if __name__ == "__main__":
    main()