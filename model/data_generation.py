"""
data_generation.py  —  Local training data generation (free, no API keys)
==========================================================================
Uses HuggingFace pipeline("text-generation") to generate training data from
your course material chunks.  Models are downloaded once to
~/.cache/huggingface and reused on subsequent runs.

TASK TYPES (exactly 4, matching your app's 4 modes)
────────────────────────────────────────────────────
  explanation   Student asks to explain a concept.   Prefix: <|explain|>
  quiz          Multiple-choice quiz generation.     Prefix: <|quiz|>
  review        Long-answer question + evaluation.   Prefix: <|review|>
  flashcard     Concise term → definition card.      Prefix: <|flashcard|>

Each example's "text" field starts with one of those 4 prefix tokens so
the model always knows which task it's being asked to do.  The vocab_size
in TransformerConfig must be set to 50_261 (50_257 + 4 special tokens).

RESUME BUG FIX
──────────────
The old code tracked resume state by chapter source path, so all chunks from
the same chapter were skipped after just one was processed.  This version
tracks by SHA-1 hash of the chunk text — each chunk is its own unit.

USAGE
─────
  # Test run, 10 chunks:
  python data_generation.py --chunks chunks.json --output data/train.jsonl \\
      --max-chunks 10

  # RTX 3090 — full quality, fp16:
  python data_generation.py --chunks chunks.json --output data/train.jsonl \\
      --model Qwen/Qwen2.5-7B-Instruct --max-chunks 300

  # RTX 3090 — 4-bit (saves VRAM, still excellent quality):
  python data_generation.py --chunks chunks.json --output data/train.jsonl \\
      --model Qwen/Qwen2.5-7B-Instruct:bnb4 --max-chunks 300

  # Resume an interrupted run (skips already-processed chunks by content hash):
  python data_generation.py --chunks chunks.json --output data/train.jsonl \\
      --model Qwen/Qwen2.5-7B-Instruct --max-chunks 500

  # Validate and preview existing output:
  python data_generation.py --validate --output data/train.jsonl

REQUIREMENTS
────────────
  pip install transformers accelerate
  pip install bitsandbytes   # only needed for :bnb4 models
"""

import hashlib
import json
import random
import argparse
import time
import warnings
from pathlib import Path
from typing import Optional

import torch

warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*generation_config.*generation-related.*")


# ─────────────────────────────────────────────────────────────────────────────
# Task prefix tokens
# These must match the special tokens added to your tokenizer/vocab.
# In TransformerConfig: vocab_size = 50_261  (50_257 + 4)
# ─────────────────────────────────────────────────────────────────────────────

TASK_PREFIX = {
    "explanation": "<|explain|>",
    "quiz":        "<|quiz|>",
    "review":      "<|review|>",
    "flashcard":   "<|flashcard|>",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────────────────────────────────────

CPU_MODEL        = "Qwen/Qwen2.5-1.5B-Instruct"
LAPTOP_GPU_MODEL = "Qwen/Qwen2.5-3B-Instruct"
HIGHEND_GPU_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

_pipeline = None


def load_generator(model_id: str):
    """
    Load a text-generation pipeline.
    Append ":bnb4" to model_id for 4-bit quantization.
    """
    from transformers import (pipeline, AutoModelForCausalLM,
                              AutoTokenizer, BitsAndBytesConfig, GenerationConfig)
    import logging
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

    use_gpu = torch.cuda.is_available()

    if model_id.endswith(":bnb4"):
        real_id = model_id[:-5]
        try:
            print(f"Loading {real_id} (4-bit bitsandbytes) on GPU...")
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(real_id)
            model = AutoModelForCausalLM.from_pretrained(
                real_id, quantization_config=bnb_cfg, device_map="auto"
            )
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            if hasattr(pipe.model, "generation_config"):
                pipe.model.generation_config.max_length = None
            return pipe
        except ImportError:
            print("    [WARN] bitsandbytes not installed — falling back to fp16")
            model_id = real_id

    device = 0 if use_gpu else -1
    dtype  = torch.float16 if use_gpu else torch.float32
    print(f"Loading {model_id} on {'GPU' if use_gpu else 'CPU'}...")
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
    )
    pipe.model.generation_config = GenerationConfig(
        max_length=None, do_sample=True, temperature=0.7, top_p=0.9,
    )
    return pipe


def call_generator(prompt: str, max_new_tokens: int = 600) -> Optional[str]:
    """Run the loaded pipeline on a prompt. Returns generated text or None."""
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Call load_generator() before call_generator().")
    try:
        result = _pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
        )
        return result[0]["generated_text"].strip()
    except Exception as e:
        print(f"    Generator error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates — one per task type, greatly improved clarity
# ─────────────────────────────────────────────────────────────────────────────

EXPLAIN_PROMPT = """You are creating training data for a teaching assistant AI.
Read the lecture material below and create ONE explanation example.

Lecture material:
{text}

Task: A student wants a concept explained in plain language.

Rules:
- The request must ask to explain ONE specific concept from the text.
- The explanation must be 3-5 sentences, written as if speaking to a student.
- Do NOT use bullet points. Write flowing prose.
- Output ONLY a JSON object. No markdown, no backticks, no extra text.

Output format:
{{"request": "Can you explain [specific concept from the text]?", "explanation": "..."}}"""


FLASHCARD_PROMPT = """You are creating training data for a teaching assistant AI.
Read the lecture material below and create ONE flashcard.

Lecture material:
{text}

A flashcard has a short FRONT (a term, concept name, or question) and a concise
BACK (the definition or answer — 1-3 sentences maximum).

Rules:
- The front should be a single term or short question, not a full sentence.
- The back must be concise and self-contained. No waffle.
- Choose something a student would genuinely want to memorise.
- Output ONLY a JSON object. No markdown, no backticks, no extra text.

Output format:
{{"front": "...", "back": "..."}}"""


ANSWER_REVIEW_PROMPT = """You are creating training data for a teaching assistant AI.
Read the lecture material below and create a long-answer question + student review.

Lecture material:
{text}

You will create:
1. A question that requires a paragraph-length answer (not yes/no, not multiple choice).
2. A reference answer (2-4 sentences, the ideal response).
3. A student answer that is PARTIALLY correct but missing 1-2 key points.
4. A constructive review of the student answer.

Rules:
- The review score must be an integer from 1 to 5.
- The feedback must say what the student got right AND specifically what they missed.
- Format: "You correctly [X] but missed [Y]. [Brief explanation of Y]."
- Output ONLY a JSON object. No markdown, no backticks, no extra text.

Output format:
{{"question": "...", "reference_answer": "...", "student_answer": "...", "review": {{"score": 3, "feedback": "You correctly stated X but missed Y because Z."}}}}"""


QUIZ_GENERATION_PROMPT = """You are creating training data for a teaching assistant AI.
Read the lecture material below and create ONE multiple-choice quiz question.

Lecture material:
{text}

Rules:
- The question must test understanding, not just recall.
- Always place the CORRECT answer at index 0 (first in the options list).
- The 3 wrong options must be plausible but clearly incorrect to someone who knows the material.
- Include a brief explanation of WHY the correct answer is right.
- Output ONLY a JSON object. No markdown, no backticks, no extra text.

Output format:
{{"question": "...", "options": ["Correct answer", "Wrong 1", "Wrong 2", "Wrong 3"], "correct_index": 0, "explanation": "..."}}"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict | list]:
    """Try several strategies to parse JSON from model output."""
    if not text:
        return None

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip ```json / ``` fences
    for fence in ["```json", "```JSON", "```"]:
        if fence in text:
            for part in text.split(fence):
                part = part.strip().rstrip("`").strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

    # Strategy 3: find the outermost { } or [ ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end   = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Format functions — produce the "text" field stored in each JSONL record.
# Each example starts with the task prefix token so the model always knows
# which task it's supposed to perform.
# ─────────────────────────────────────────────────────────────────────────────

def format_explanation(request: str, explanation: str, context: str = "") -> str:
    prefix = TASK_PREFIX["explanation"]
    if context:
        return f"{prefix}\nContext: {context}\nRequest: {request}\nResponse: {explanation}<|endoftext|>"
    return f"{prefix}\nRequest: {request}\nResponse: {explanation}<|endoftext|>"


def format_flashcard(front: str, back: str, context: str = "") -> str:
    prefix = TASK_PREFIX["flashcard"]
    if context:
        return f"{prefix}\nContext: {context}\nFront: {front}\nBack: {back}<|endoftext|>"
    return f"{prefix}\nFront: {front}\nBack: {back}<|endoftext|>"


def format_review(question: str, student_answer: str, review_text: str, context: str = "") -> str:
    prefix = TASK_PREFIX["review"]
    if context:
        return (f"{prefix}\nContext: {context}\n"
                f"Question: {question}\n"
                f"Student answer: {student_answer}\n"
                f"Review: {review_text}<|endoftext|>")
    return (f"{prefix}\n"
            f"Question: {question}\n"
            f"Student answer: {student_answer}\n"
            f"Review: {review_text}<|endoftext|>")


def format_quiz(quiz: dict, context: str = "") -> str:
    prefix = TASK_PREFIX["quiz"]
    quiz_str = json.dumps(quiz, ensure_ascii=False)
    if context:
        return f"{prefix}\nContext: {context}\nQuiz: {quiz_str}<|endoftext|>"
    return f"{prefix}\nQuiz: {quiz_str}<|endoftext|>"


# ─────────────────────────────────────────────────────────────────────────────
# Context quality filter
# ─────────────────────────────────────────────────────────────────────────────

def _is_good_chunk(text: str, min_words: int = 40) -> bool:
    """
    Return False for chunks that are too short or end mid-sentence.
    These produce confusing training examples.
    """
    words = text.split()
    if len(words) < min_words:
        return False
    # Truncated if the last non-whitespace character is not sentence-ending
    last_char = text.rstrip()[-1] if text.rstrip() else ""
    if last_char not in ".?!\"'":
        # Allow if it ends with a closing bracket / code block — still valid
        if last_char not in ")]}`":
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Chunk hashing — used for reliable resume tracking
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_hash(chunk: dict) -> str:
    """
    SHA-1 of the chunk text — unique per chunk regardless of source file.

    The old code tracked by source file path, so all chunks from the same
    chapter were skipped after the first one was processed.  This fixes that.
    """
    text = chunk.get("text", "")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_modes_local(
    chunks_json_path: str,
    output_jsonl_path: str,
    model_id: str,
    max_chunks: int = 200,
    n_explain:   int = 1,
    n_flashcard: int = 1,
    n_review:    int = 1,
    n_quiz:      int = 1,
    save_every:  int = 20,
    min_chunk_words: int = 40,
):
    """
    Generate training examples from course material chunks.

    For each accepted chunk this produces (up to):
      n_explain   × explanation examples
      n_flashcard × flashcard examples
      n_review    × answer-review examples
      n_quiz      × quiz examples

    Resume: already-processed chunks are identified by content hash and skipped.
    Running with a larger --max-chunks will process the additional chunks.
    """
    global _pipeline

    with open(chunks_json_path, encoding="utf-8") as f:
        all_chunks = json.load(f)

    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume: load existing results and collect processed chunk hashes ──────
    results: list = []
    processed_hashes: set = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    results.append(ex)
                    if "chunk_hash" in ex:
                        processed_hashes.add(ex["chunk_hash"])
        print(f"Resuming: {len(results)} existing examples "
              f"({len(processed_hashes)} processed chunks)")

    # ── Filter out low-quality chunks before sampling ─────────────────────────
    good_chunks = [c for c in all_chunks if _is_good_chunk(c.get("text", ""), min_chunk_words)]
    bad_count   = len(all_chunks) - len(good_chunks)
    if bad_count:
        print(f"Filtered {bad_count} low-quality / truncated chunks "
              f"({len(good_chunks)} remaining)")

    # ── Sample up to max_chunks, then remove already-processed ones ───────────
    # Shuffle deterministically so reruns with larger max_chunks add NEW chunks
    # rather than re-sampling the same ones.
    rng = random.Random(42)
    shuffled = good_chunks[:]
    rng.shuffle(shuffled)

    selected  = shuffled[:max_chunks]
    remaining = [c for c in selected if _chunk_hash(c) not in processed_hashes]

    print(f"Selected {len(selected)} chunks (max_chunks={max_chunks}), "
          f"{len(remaining)} not yet processed.")

    if not remaining:
        print("Nothing new to process. "
              "Try --max-chunks with a larger value to process more chunks.")
        return results

    # ── Load generator once ───────────────────────────────────────────────────
    _pipeline = load_generator(model_id)

    skipped = 0

    for i, chunk in enumerate(remaining):
        text       = chunk.get("text", "")
        breadcrumb = chunk.get("metadata", {}).get("breadcrumb", "")
        source     = chunk.get("source", f"chunk_{i}")
        topic      = breadcrumb or source
        chunk_id   = _chunk_hash(chunk)

        # Truncate context for the format functions (keeps prompts from growing huge)
        context_snippet = text[:300].strip()

        print(f"  [{i+1}/{len(remaining)}] {topic[:100]}")
        t0 = time.time()

        # ── Explanation ───────────────────────────────────────────────────────
        for _ in range(n_explain):
            raw  = call_generator(EXPLAIN_PROMPT.format(text=text[:700]))
            data = extract_json(raw)
            if isinstance(data, dict) and "request" in data and "explanation" in data:
                req = str(data["request"]).strip()
                exp = str(data["explanation"]).strip()
                if len(req) > 10 and len(exp) > 30:
                    results.append({
                        "type":        "explanation",
                        "text":        format_explanation(req, exp, context=context_snippet),
                        "request":     req,
                        "explanation": exp,
                        "source":      source,
                        "topic":       topic,
                        "chunk_hash":  chunk_id,
                    })
            else:
                skipped += 1

        # ── Flashcard ─────────────────────────────────────────────────────────
        for _ in range(n_flashcard):
            raw  = call_generator(FLASHCARD_PROMPT.format(text=text[:700]))
            data = extract_json(raw)
            if isinstance(data, dict) and "front" in data and "back" in data:
                front = str(data["front"]).strip()
                back  = str(data["back"]).strip()
                if len(front) > 3 and len(back) > 10:
                    results.append({
                        "type":       "flashcard",
                        "text":       format_flashcard(front, back, context=context_snippet),
                        "front":      front,
                        "back":       back,
                        "source":     source,
                        "topic":      topic,
                        "chunk_hash": chunk_id,
                    })
            else:
                skipped += 1

        # ── Answer review ─────────────────────────────────────────────────────
        for _ in range(n_review):
            raw  = call_generator(ANSWER_REVIEW_PROMPT.format(text=text[:700]))
            data = extract_json(raw)
            if isinstance(data, dict) and all(k in data for k in ["question", "student_answer", "review"]):
                review      = data["review"]
                score       = review.get("score", "?")
                feedback    = str(review.get("feedback", "")).strip()
                if feedback and len(feedback) > 20:
                    # Standardise to "Score: X/5. Feedback..." format
                    review_text = f"Score: {score}/5. {feedback}"
                    results.append({
                        "type":           "review",
                        "text":           format_review(
                                              str(data["question"]).strip(),
                                              str(data["student_answer"]).strip(),
                                              review_text,
                                              context=context_snippet,
                                          ),
                        "question":       str(data["question"]).strip(),
                        "student_answer": str(data["student_answer"]).strip(),
                        "review":         review_text,
                        "reference":      str(data.get("reference_answer", "")).strip(),
                        "source":         source,
                        "topic":          topic,
                        "chunk_hash":     chunk_id,
                    })
            else:
                skipped += 1

        # ── Quiz ──────────────────────────────────────────────────────────────
        for _ in range(n_quiz):
            raw  = call_generator(QUIZ_GENERATION_PROMPT.format(text=text[:700]))
            data = extract_json(raw)
            # Accept either the old nested format or the new flat format
            if isinstance(data, dict) and "quiz_output" in data:
                data = data["quiz_output"]   # unwrap old format if present
            if (isinstance(data, dict)
                    and isinstance(data.get("options"), list)
                    and len(data["options"]) >= 2
                    and "question" in data):
                # Correct answer is always at index 0 during generation.
                # Shuffle so it's not always first at inference time.
                correct_answer = data["options"][data.get("correct_index", 0)]
                opts = data["options"][:]
                random.shuffle(opts)
                data["options"]       = opts
                data["correct_index"] = opts.index(correct_answer)

                results.append({
                    "type":       "quiz",
                    "text":       format_quiz(data, context=context_snippet),
                    "quiz":       data,
                    "source":     source,
                    "topic":      topic,
                    "chunk_hash": chunk_id,
                })
            else:
                skipped += 1

        elapsed = time.time() - t0
        print(f"    {elapsed:.1f}s  running total: {len(results)}")

        if (i + 1) % save_every == 0:
            _save(results, output_path)
            print(f"  [checkpoint] saved {len(results)} examples")

    _save(results, output_path)

    counts: dict = {}
    for r in results:
        counts[r["type"]] = counts.get(r["type"], 0) + 1

    print(f"\n{'='*55}")
    print(f"Done. {len(results)} total examples ({skipped} skipped as malformed):")
    for t in ["explanation", "flashcard", "review", "quiz"]:
        print(f"  {t:12s}  {counts.get(t, 0):5d}")
    other = {k: v for k, v in counts.items() if k not in ["explanation", "flashcard", "review", "quiz"]}
    for t, n in other.items():
        print(f"  {t:12s}  {n:5d}  (legacy — consider removing)")
    print(f"Saved to {output_jsonl_path}")
    return results


def _save(results: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(jsonl_path: str):
    """Print a summary and random samples from a generated JSONL file."""
    by_type: dict = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            by_type.setdefault(ex["type"], []).append(ex)

    total = sum(len(v) for v in by_type.values())
    print(f"\nDataset: {total} total examples")

    expected_types = ["explanation", "flashcard", "review", "quiz"]
    for t in expected_types:
        examples = by_type.get(t, [])
        if not examples:
            print(f"\n{'─'*60}\n⚠ TYPE '{t}' has 0 examples — check your generation run")
            continue
        # Check task prefix presence
        has_prefix = sum(1 for e in examples if e["text"].startswith(TASK_PREFIX[t]))
        print(f"\n{'─'*60}\nTYPE: {t}  ({len(examples)} examples, "
              f"{has_prefix}/{len(examples)} have task prefix)")
        for ex in random.sample(examples, min(2, len(examples))):
            print(ex["text"][:500])
            print("...")

    # Warn about legacy types
    for t in by_type:
        if t not in expected_types:
            print(f"\n⚠ Legacy type '{t}' found ({len(by_type[t])} examples). "
                  f"Consider regenerating or converting these.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    default_model = LAPTOP_GPU_MODEL if use_gpu else CPU_MODEL

    parser = argparse.ArgumentParser(
        description="Generate training data using a local HuggingFace model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Model presets (pass to --model):
  CPU / slow machine : {CPU_MODEL}
  Laptop GPU         : {LAPTOP_GPU_MODEL}   (default when GPU detected)
  RTX 3090 fp16      : {HIGHEND_GPU_MODEL}
  RTX 3090 4-bit     : {HIGHEND_GPU_MODEL}:bnb4  (needs bitsandbytes)

Task prefix tokens added to vocab (set vocab_size=50261 in TransformerConfig):
  <|explain|>   <|quiz|>   <|review|>   <|flashcard|>
        """,
    )
    parser.add_argument("--chunks",         default="../rag/rag_index/chunks.json",
                        help="Path to chunks.json from your chunker")
    parser.add_argument("--output",         default="data/train.jsonl",
                        help="Output JSONL path (also used as resume checkpoint)")
    parser.add_argument("--model",          default=default_model,
                        help="HuggingFace model ID, optionally with :bnb4 suffix")
    parser.add_argument("--max-chunks",     type=int, default=200,
                        help="Max number of chunks to process (increase to add more data)")
    parser.add_argument("--n-explain",      type=int, default=1,
                        help="Explanation examples per chunk")
    parser.add_argument("--n-flashcard",    type=int, default=1,
                        help="Flashcard examples per chunk")
    parser.add_argument("--n-review",       type=int, default=1,
                        help="Answer-review examples per chunk")
    parser.add_argument("--n-quiz",         type=int, default=1,
                        help="Quiz examples per chunk")
    parser.add_argument("--min-chunk-words",type=int, default=40,
                        help="Skip chunks shorter than this many words (filters truncated chunks)")
    parser.add_argument("--validate",       action="store_true",
                        help="Validate and preview an existing output file instead of generating")
    args = parser.parse_args()

    print(f"Device : {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"Model  : {args.model}")

    if args.validate:
        validate_dataset(args.output)
    else:
        generate_all_modes_local(
            chunks_json_path=args.chunks,
            output_jsonl_path=args.output,
            model_id=args.model,
            max_chunks=args.max_chunks,
            n_explain=args.n_explain,
            n_flashcard=args.n_flashcard,
            n_review=args.n_review,
            n_quiz=args.n_quiz,
            min_chunk_words=args.min_chunk_words,
        )