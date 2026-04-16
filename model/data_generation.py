"""
data_generation.py  —  Local training data generation (free, no API keys)
==========================================================================
Uses HuggingFace pipeline("text-generation") to generate training data from
your course material chunks.  Models are downloaded once to
~/.cache/huggingface and reused on subsequent runs.

MODEL TIERS
───────────
  CPU        Qwen/Qwen2.5-1.5B-Instruct          ~3 GB  slow, works anywhere
  Laptop GPU Qwen/Qwen2.5-3B-Instruct            ~6 GB  good for testing
  RTX 3090   Qwen/Qwen2.5-7B-Instruct            ~14 GB fp16, best quality
  RTX 3090   Qwen/Qwen2.5-7B-Instruct:bnb4       ~5 GB  4-bit, fastest on 3090

  Append ":bnb4" to ANY model ID for 4-bit loading (needs bitsandbytes):
    "mistralai/Mistral-7B-Instruct-v0.3:bnb4"    ~5 GB  alternative 7B

  The 7B models produce noticeably better explanations and review feedback
  than the 3B models — use them on the 3090 for your real data generation runs.

USAGE
─────
  # Laptop GPU (test run, 10 chunks):
  python data_generation.py --chunks chunks.json --output data/train.jsonl \\
      --max-chunks 10

  # RTX 3090 — full quality, fp16:
  python data_generation.py --chunks chunks.json --output data/train.jsonl \\
      --model Qwen/Qwen2.5-7B-Instruct --max-chunks 300

  # RTX 3090 — 4-bit (saves VRAM, still excellent quality):
  python data_generation.py --chunks chunks.json --output data/train.jsonl \\
      --model Qwen/Qwen2.5-7B-Instruct:bnb4 --max-chunks 300

  # Resume interrupted run (automatically skips already-processed chunks):
  python data_generation.py --chunks chunks.json --output data/train.jsonl \\
      --model Qwen/Qwen2.5-7B-Instruct

  # Validate existing output:
  python data_generation.py --validate --output data/train.jsonl

REQUIREMENTS
────────────
  pip install transformers accelerate
  pip install bitsandbytes   # only needed for :bnb4 models
"""

import json
import random
import argparse
import time
import warnings
from pathlib import Path
from typing import Optional

import torch

# Suppress harmless transformers generation warnings caused by some models
# shipping a generation_config.json with max_length=20 baked in.
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*generation_config.*generation-related.*")


# ─────────────────────────────────────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────────────────────────────────────

CPU_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"   # ~3 GB, runs on CPU
LAPTOP_GPU_MODEL = "Qwen/Qwen2.5-3B-Instruct"    # ~6 GB, good for laptop GPU testing
HIGHEND_GPU_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # ~14 GB fp16, recommended for RTX 3090
                                                  # Use ":bnb4" suffix to cut to ~5 GB 4-bit


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

_pipeline = None   # module-level cache so we only load once


def load_generator(model_id: str):
    """
    Load a text-generation pipeline from HuggingFace.
    Models download to ~/.cache/huggingface on first use, then reused.

    Append ":bnb4" to model_id for 4-bit quantization, e.g.:
        "Qwen/Qwen2.5-7B-Instruct:bnb4"          -- recommended for RTX 3090
        "mistralai/Mistral-7B-Instruct-v0.3:bnb4" -- alternative
    """
    from transformers import (pipeline, AutoModelForCausalLM,
                              AutoTokenizer, BitsAndBytesConfig, GenerationConfig)
    import logging
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

    use_gpu = torch.cuda.is_available()

    # ":bnb4" suffix triggers 4-bit bitsandbytes quantization
    if model_id.endswith(":bnb4"):
        real_id = model_id[:-5]
        try:
            print(f"Loading {real_id} (4-bit bitsandbytes) on GPU...")
            print("    (downloaded once to ~/.cache/huggingface)")
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
            print("    [WARN] bitsandbytes not installed — pip install bitsandbytes")
            print("    [WARN] falling back to standard fp16 loading")
            model_id = real_id

    device = 0 if use_gpu else -1
    dtype  = torch.float16 if use_gpu else torch.float32
    print(f"Loading {model_id} on {'GPU' if use_gpu else 'CPU'}...")
    print("    (downloaded once to ~/.cache/huggingface)")
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
    )
    # Overwrite stored generation_config to avoid max_length conflicts from
    # some model repos that ship a generation_config.json with max_length=20.
    pipe.model.generation_config = GenerationConfig(
        max_length=None,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return pipe


def call_generator(prompt: str, max_new_tokens: int = 512) -> Optional[str]:
    """Run the loaded pipeline on a prompt.  Returns generated text or None."""
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
# Prompt templates
# (explicit JSON instructions because instruction-tuned models need them)
# ─────────────────────────────────────────────────────────────────────────────

EXPLAIN_PROMPT = """You are creating training data for a teaching assistant.
Read the lecture material below and write ONE explanation example.

Lecture material:
{text}

Rules:
- Write a student request asking to explain something specific from the text.
- Write a 3-5 sentence explanation as if talking to a student.
- Output ONLY a JSON object. No markdown, no backticks, no extra text.

Output format:
{{"request": "Can you explain [concept]?", "explanation": "..."}}"""


ANSWER_REVIEW_PROMPT = """You are creating training data for a teaching assistant.
Read the lecture material below and create a question, a weak student answer, and a review.

Lecture material:
{text}

Rules:
- The student answer should be partially correct but missing 1-2 key points.
- The review score must be an integer 1-5.
- Output ONLY a JSON object. No markdown, no backticks, no extra text.

Output format:
{{"question": "...", "reference_answer": "...", "student_answer": "...", "review": {{"score": 3, "feedback": "You correctly stated X but missed Y because Z."}}}}"""


QUIZ_GENERATION_PROMPT = """You are creating training data for a teaching assistant.
Read the lecture material below and create a multiple-choice quiz question.

Lecture material:
{text}

Rules:
- correct_index must be 0 (correct answer is always options[0] during generation).
- The other 3 options must be plausible but wrong.
- Output ONLY a JSON object. No markdown, no backticks, no extra text.

Output format:
{{"generate_request": "Generate a question about [concept].", "quiz_output": {{"question": "...", "options": ["Correct answer", "Wrong 1", "Wrong 2", "Wrong 3"], "correct_index": 0, "explanation": "..."}}}}"""


QA_PROMPT = """You are creating training data for a teaching assistant.
Read the lecture material below and create {n} question-answer pairs.

Lecture material:
{text}

Rules:
- Questions should be specific and answerable from the text.
- Answers should be 1-3 sentences.
- Output ONLY a JSON array. No markdown, no backticks, no extra text.

Output format:
[{{"question": "...", "answer": "..."}}, ...]"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction — models sometimes wrap output in markdown fences
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict | list]:
    if not text:
        return None

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip ```json ... ``` fences
    for fence in ["```json", "```JSON", "```"]:
        if fence in text:
            for part in text.split(fence):
                part = part.strip().rstrip("`").strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

    # Strategy 3: find first { or [
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
# Format functions
# ─────────────────────────────────────────────────────────────────────────────

def format_explanation_example(request: str, explanation: str, context: str = "") -> str:
    if context:
        return f"Context: {context}\nRequest: {request}\nResponse: {explanation}<|endoftext|>"
    return f"Request: {request}\nResponse: {explanation}<|endoftext|>"


def format_review_example(question: str, student_answer: str, review_text: str, context: str = "") -> str:
    if context:
        return (f"Context: {context}\n"
                f"Question: {question}\n"
                f"Student answer: {student_answer}\n"
                f"Review: {review_text}<|endoftext|>")
    return (f"Question: {question}\n"
            f"Student answer: {student_answer}\n"
            f"Review: {review_text}<|endoftext|>")


def format_qa_example(question: str, answer: str, context: str = "") -> str:
    if context:
        return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}<|endoftext|>"
    return f"Question: {question}\nAnswer: {answer}<|endoftext|>"


def format_quiz_example(generate_request: str, quiz_output: dict, context: str = "") -> str:
    quiz_str = json.dumps(quiz_output, ensure_ascii=False)
    if context:
        return (f"Context: {context}\n"
                f"Request: {generate_request}\n"
                f"Quiz: {quiz_str}<|endoftext|>")
    return f"Request: {generate_request}\nQuiz: {quiz_str}<|endoftext|>"


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_modes_local(
    chunks_json_path: str,
    output_jsonl_path: str,
    model_id: str,
    max_chunks: int = 200,
    n_qa: int = 3,
    n_explain: int = 1,
    n_review: int = 1,
    n_quiz_gen: int = 1,
    save_every: int = 20,
):
    """
    Generate training examples from course material chunks.

    For each chunk this produces (up to):
      - n_qa        QA pairs
      - n_explain   explanation examples
      - n_review    answer-review examples
      - n_quiz_gen  quiz generation examples

    Supports resuming: if output_jsonl_path already exists, already-processed
    chunk sources are skipped.

    The output JSONL is consumed by dataset.py's load_local_generated_data().
    """
    global _pipeline

    with open(chunks_json_path, encoding="utf-8") as f:
        chunks = json.load(f)

    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip chunks we've already processed
    results = []
    processed_sources: set = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    results.append(ex)
                    processed_sources.add(ex.get("source", ""))
        print(f"Resuming: {len(results)} existing examples loaded")

    selected  = random.sample(chunks, min(max_chunks, len(chunks)))
    remaining = [c for c in selected if c.get("source", "") not in processed_sources]
    print(f"Processing {len(remaining)} chunks "
          f"(skipping {len(selected) - len(remaining)} already done)")

    # Load model once
    _pipeline = load_generator(model_id)

    skipped = 0

    for i, chunk in enumerate(remaining):
        text       = chunk["text"][:700]
        breadcrumb = chunk.get("metadata", {}).get("breadcrumb", "")
        source     = chunk.get("source", f"chunk_{i}")
        topic      = breadcrumb or "the material"

        print(f"  [{i+1}/{len(remaining)}] {topic[:110]}")
        t0 = time.time()

        # ── QA pairs ──────────────────────────────────────────────────────────
        raw  = call_generator(QA_PROMPT.format(text=text, n=n_qa))
        data = extract_json(raw)
        if data:
            pairs = data if isinstance(data, list) else data.get("pairs", data.get("questions", []))
            for pair in pairs[:n_qa]:
                if "question" in pair and "answer" in pair:
                    q = str(pair["question"]).strip()
                    a = str(pair["answer"]).strip()
                    if len(q) > 10 and len(a) > 10:
                        results.append({
                            "type":     "qa",
                            "text":     format_qa_example(q, a, context=text[:300]),
                            "question": q, "answer": a,
                            "source":   source, "topic": topic,
                        })

        # ── Explanation ───────────────────────────────────────────────────────
        raw  = call_generator(EXPLAIN_PROMPT.format(text=text))
        data = extract_json(raw)
        if data and "request" in data and "explanation" in data:
            req = str(data["request"]).strip()
            exp = str(data["explanation"]).strip()
            if len(req) > 10 and len(exp) > 30:
                results.append({
                    "type":        "explanation",
                    "text":        format_explanation_example(req, exp, context=text[:300]),
                    "request":     req, "explanation": exp,
                    "source":      source, "topic": topic,
                })
        else:
            skipped += 1

        # ── Answer review ─────────────────────────────────────────────────────
        raw  = call_generator(ANSWER_REVIEW_PROMPT.format(text=text))
        data = extract_json(raw)
        if data and all(k in data for k in ["question", "student_answer", "review"]):
            review   = data["review"]
            feedback = str(review.get("feedback", "")).strip()
            if feedback and len(feedback) > 20:
                review_text = f"Score: {review.get('score', '?')}/5. {feedback}"
                results.append({
                    "type":           "review",
                    "text":           format_review_example(
                                          str(data["question"]).strip(),
                                          str(data["student_answer"]).strip(),
                                          review_text,
                                          context=text[:300],
                                      ),
                    "question":       str(data["question"]).strip(),
                    "student_answer": str(data["student_answer"]).strip(),
                    "review":         review_text,
                    "reference":      str(data.get("reference_answer", "")).strip(),
                    "source":         source, "topic": topic,
                })
        else:
            skipped += 1

        # ── Quiz generation ───────────────────────────────────────────────────
        raw  = call_generator(QUIZ_GENERATION_PROMPT.format(text=text))
        data = extract_json(raw)
        if data and "generate_request" in data and "quiz_output" in data:
            quiz = data["quiz_output"]
            if (isinstance(quiz.get("options"), list)
                    and len(quiz["options"]) >= 2
                    and "question" in quiz):
                # Shuffle so correct answer isn't always index 0 at inference
                correct_answer  = quiz["options"][quiz.get("correct_index", 0)]
                opts            = quiz["options"][:]
                random.shuffle(opts)
                quiz["options"]        = opts
                quiz["correct_index"]  = opts.index(correct_answer)

                results.append({
                    "type":             "quiz_gen",
                    "text":             format_quiz_example(
                                            str(data["generate_request"]).strip(),
                                            quiz,
                                            context=text[:300],
                                        ),
                    "generate_request": str(data["generate_request"]).strip(),
                    "quiz_output":      quiz,
                    "source":           source, "topic": topic,
                })
        else:
            skipped += 1

        elapsed = time.time() - t0
        print(f"    {elapsed:.1f}s  total: {len(results)}")

        if (i + 1) % save_every == 0:
            _save(results, output_path)
            print(f"  [checkpoint] saved {len(results)} examples")

    _save(results, output_path)

    counts: dict = {}
    for r in results:
        counts[r["type"]] = counts.get(r["type"], 0) + 1

    print(f"\n{'='*50}")
    print(f"Generated {len(results)} total ({skipped} skipped as malformed):")
    for t, n in sorted(counts.items()):
        print(f"  {t:12s} {n:5d}")
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

    print(f"\nDataset: {sum(len(v) for v in by_type.values())} total examples")
    for t, examples in by_type.items():
        print(f"\n{'─'*60}\nTYPE: {t}  ({len(examples)} examples)")
        for ex in random.sample(examples, min(2, len(examples))):
            print(ex["text"][:400])
            print("...")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()

    # Auto-select a sensible default: GPU → laptop model, CPU → cpu model.
    # Pass --model explicitly to use the 3090 high-quality model.
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
        """,
    )
    parser.add_argument("--chunks",      default="../rag/rag_index/chunks.json",
                        help="Path to chunks.json from your chunker")
    parser.add_argument("--output",      default="data/train.jsonl",
                        help="Output JSONL path (also used as resume checkpoint)")
    parser.add_argument("--model",       default=default_model,
                        help="HuggingFace model ID, optionally with :bnb4 suffix")
    parser.add_argument("--max-chunks",  type=int, default=200,
                        help="Max number of chunks to process")
    parser.add_argument("--n-qa",        type=int, default=3,
                        help="QA pairs per chunk")
    parser.add_argument("--n-explain",   type=int, default=1,
                        help="Explanation examples per chunk")
    parser.add_argument("--n-review",    type=int, default=1,
                        help="Answer-review examples per chunk")
    parser.add_argument("--n-quiz",      type=int, default=1,
                        help="Quiz generation examples per chunk")
    parser.add_argument("--validate",    action="store_true",
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
            n_qa=args.n_qa,
            n_explain=args.n_explain,
            n_review=args.n_review,
            n_quiz_gen=args.n_quiz,
        )