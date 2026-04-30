#!/usr/bin/env python3
"""
run_exam_eval.py  —  Evaluate one model against one or all exams in a folder
=============================================================================
Two backends:
  hf      — any HuggingFace CausalLM, with optional LoRA adapter
  custom  — your TeachingAssistantModel (.pt checkpoint)

Optional RAG:
  --use-rag   prepends retrieved lecture chunks to every question prompt.
              Uses rag/rag_index/ (auto-resolved from workspace root).

Usage:
  # Single exam, HF base model, no judge:
  python run_exam_eval.py --exam exam/DAT255_V25.json \
      --backend hf --model-id Qwen/Qwen2.5-1.5B-Instruct --no-judge

  # All exams, LoRA adapter, local judge, with RAG:
  python run_exam_eval.py --exam-dir exam/ \
      --backend hf --model-id Qwen/Qwen2.5-1.5B-Instruct \
      --lora-path ../model/checkpoints/qwen_lora \
      --judge-model-id Qwen/Qwen2.5-7B-Instruct \
      --use-rag --out-dir results/qwen_lora_rag

  # Custom scratch transformer:
  python run_exam_eval.py --exam-dir exam/ \
      --backend custom --checkpoint ../model/checkpoints/best.pt \
      --out-dir results/custom_25M
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

TASK_PREFIX = {
    "multiple_choice": "<|quiz|>",
    "open_ended":      "<|explain|>",
}


def build_mc_prompt(q: dict, for_custom: bool = False,
                    context: str = "") -> str:
    num = q.get("num_answers", 1)
    options_text = ""
    if q.get("options"):
        options_text = "\n" + "\n".join(f"{k}) {v}" for k, v in q["options"].items())

    instruction = (
        f"Select the {num} correct answers. "
        f"Reply with only the {num} letters separated by spaces, e.g. 'A C'."
        if num > 1 else
        "Select the correct answer. Reply with only the single letter (A, B, C, D, or E)."
    )

    if for_custom:
        ctx_block = f"Context: {context}\n\n" if context else ""
        return (
            f"{TASK_PREFIX['multiple_choice']} {ctx_block}"
            f"{q['question_text']}{options_text}\n\n{instruction}\nAnswer:"
        )
    ctx_block = f"Context:\n{context}\n\n" if context else ""
    return f"{ctx_block}{q['question_text']}{options_text}\n\n{instruction}"


def build_open_prompt(q: dict, for_custom: bool = False,
                      context: str = "") -> str:
    word_limit = q["points"] * 100
    if for_custom:
        ctx_block = f"Context: {context}\n\n" if context else ""
        return (
            f"{TASK_PREFIX['open_ended']} {ctx_block}"
            f"{q['question_text']}\n\nWrite a clear and concise answer. "
            f"Maximum {word_limit} words.\nAnswer:"
        )
    ctx_block = f"Context:\n{context}\n\n" if context else ""
    return (
        f"{ctx_block}{q['question_text']}\n\n"
        f"Write a clear and concise answer. Maximum {word_limit} words."
    )


def build_prompt(q: dict, for_custom: bool = False, context: str = "") -> str:
    if q["question_type"] == "multiple_choice":
        return build_mc_prompt(q, for_custom, context)
    return build_open_prompt(q, for_custom, context)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Model backends
# ─────────────────────────────────────────────────────────────────────────────

class HFModel:
    """
    HuggingFace CausalLM — base model or with a LoRA adapter.
    Point --lora-path at the root qwen_lora/ dir (contains adapter_config.json).
    The checkpoint-N subdirs are mid-training snapshots; use the root for final.
    """

    def __init__(self, model_id: str, lora_path: Optional[str] = None,
                 max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if lora_path:
            from peft import PeftModel
            print(f"Applying LoRA adapter: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.is_instruct = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )
        print(f"  instruct mode: {self.is_instruct}")

    def generate(self, prompt: str) -> str:
        import torch
        if self.is_instruct:
            input_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            input_text = prompt

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


class CustomModel:
    """
    Your scratch TeachingAssistantModel, loaded from a .pt checkpoint.
    Reuses load_checkpoint() and generate() the same way run_eval.py does.
    """

    def __init__(self, checkpoint_path: str, max_new_tokens: int = 200,
                 temperature: float = 0.7):
        import torch

        # eval/ lives one level below workspace root → model/ is a sibling
        eval_dir  = Path(__file__).resolve().parent
        model_dir = eval_dir.parent / "model"
        sys.path.insert(0, str(model_dir))

        from train import load_checkpoint
        from dataset import Tokenizer

        print(f"Loading custom checkpoint: {checkpoint_path}")
        model, _opt, step, metrics = load_checkpoint(checkpoint_path, "cpu")

        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.model          = model.to(self.device)
        self.model.eval()
        self.tokenizer      = Tokenizer()
        self.eot_id         = self.tokenizer.eot_id
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        print(f"  step={step}  val_loss={metrics.get('best_val_loss', 'N/A')}")

    def generate(self, prompt: str) -> str:
        import torch
        ids = self.tokenizer.encode(prompt)
        x   = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            x,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop_token=self.eot_id,
        )
        new_ids = out[0, len(ids):].tolist()
        if self.eot_id in new_ids:
            new_ids = new_ids[:new_ids.index(self.eot_id)]
        return self.tokenizer.decode(new_ids).strip()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: MC scoring
# ─────────────────────────────────────────────────────────────────────────────

def extract_letters(text: str) -> list[str]:
    return re.findall(r'\b([A-E])\b', text.upper())


def score_mc(response: str, correct: list[str], points: int,
             num_answers: int) -> dict:
    predicted = extract_letters(response)

    if num_answers == 1:
        got   = predicted[0] if predicted else ""
        score = float(points) if got == correct[0] else 0.0
        return {
            "predicted": got or response[:50],
            "correct":   correct[0],
            "score":     score,
            "max_score": points,
            "note":      "correct" if score == points else "wrong",
        }

    pred_set    = set(predicted[:num_answers * 2])
    correct_set = set(correct)
    hits  = len(pred_set & correct_set)
    wrong = len(pred_set - correct_set)
    raw   = (hits / num_answers) * points - (wrong * points / num_answers)
    score = round(max(0.0, raw), 2)
    return {
        "predicted": sorted(pred_set),
        "correct":   sorted(correct_set),
        "score":     score,
        "max_score": points,
        "note":      f"{hits}/{num_answers} correct, {wrong} wrong picks",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: LLM judge for open-ended questions
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an exam grader for a university deep learning course. "
    "Grade the student answer strictly based on the rubric provided. "
    "Respond ONLY with valid JSON: {\"score\": <int>, \"reason\": \"<one sentence>\"}"
)

_JUDGE_TMPL = """\
Question ({points}p):
{question}

Rubric:
{rubric}

Student answer:
{answer}

Award an integer score from 0 to {points}.
Return only JSON: {{"score": <int>, "reason": "<str>"}}"""


def run_judge(judge_model: HFModel, question: dict, answer: str) -> dict:
    rubric = question.get("model_answer") or "Award points for a clear and accurate answer."
    prompt = f"{_JUDGE_SYSTEM}\n\n" + _JUDGE_TMPL.format(
        points=question["points"],
        question=question["question_text"][:500],
        rubric=rubric[:800],
        answer=answer[:1000],
    )
    raw = judge_model.generate(prompt)
    try:
        m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if m:
            data  = json.loads(m.group())
            score = max(0, min(int(data.get("score", 0)), question["points"]))
            return {"score": score, "max_score": question["points"],
                    "reason": data.get("reason", ""), "raw_judge": raw[:200]}
    except (json.JSONDecodeError, ValueError):
        pass
    return {"score": 0, "max_score": question["points"],
            "reason": "judge parse failed", "raw_judge": raw[:200]}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Evaluation loop for one exam
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_exam(
    exam_path:   str,
    model,
    model_name:  str,
    is_custom:   bool = False,
    judge_model: Optional[HFModel] = None,
    retriever=None,   # RAGRetriever instance or None
) -> dict:
    with open(exam_path, encoding="utf-8") as f:
        exam = json.load(f)

    questions = exam["questions"]
    exam_name = exam["metadata"]["source_pdf"]
    use_rag   = retriever is not None

    print(f"\n{'='*60}")
    print(f"Model : {model_name}  {'[+RAG]' if use_rag else ''}")
    print(f"Exam  : {exam_name}  ({len(questions)} questions)")
    print('='*60)

    results = []
    for q in questions:
        qnum  = q["question_number"]
        qtype = q["question_type"]
        print(f"  Q{qnum:2d} ({qtype[:2].upper()}, {q['points']}p) ", end="", flush=True)

        # Retrieve context if RAG is enabled
        context = ""
        rag_chunks = []
        if use_rag:
            rag_chunks = retriever.retrieve(q["question_text"])
            context    = retriever.build_context(rag_chunks)

        prompt = build_prompt(q, for_custom=is_custom, context=context)

        try:
            response = model.generate(prompt)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "question_number": qnum, "question_type": qtype,
                "points": q["points"], "response": "",
                "score": 0, "max_score": q["points"], "error": str(e),
            })
            continue

        if qtype == "multiple_choice":
            correct = q.get("correct_answer") or []
            num_ans = q.get("num_answers", 1)
            if not correct:
                scored = {
                    "predicted": extract_letters(response) or response[:50],
                    "correct": None, "score": None,
                    "max_score": q["points"], "note": "no answer key",
                }
            else:
                scored = score_mc(response, correct, q["points"], num_ans)
            result = {"question_number": qnum, "question_type": qtype,
                      "points": q["points"], "response": response, **scored}
            print(f"pred={result.get('predicted')}  "
                  f"correct={result.get('correct')}  "
                  f"score={result.get('score')}/{q['points']}")

        else:
            if judge_model is None:
                scored = {"score": None, "max_score": q["points"],
                          "reason": "no judge"}
            else:
                scored = run_judge(judge_model, q, response)
            result = {"question_number": qnum, "question_type": qtype,
                      "points": q["points"], "response": response, **scored}
            print(f"score={result.get('score')}/{q['points']}  "
                  f"{result.get('reason', '')[:60]}")

        # Attach RAG metadata if used (useful for debugging retrieval quality)
        if use_rag:
            result["rag_chunks"] = [
                {"score": c.get("score"), "source": c.get("source"),
                 "breadcrumb": c.get("breadcrumb")}
                for c in rag_chunks
            ]

        results.append(result)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    scorable   = [r for r in results if r.get("score") is not None]
    mc_results = [r for r in scorable if r["question_type"] == "multiple_choice"]
    oe_results = [r for r in scorable if r["question_type"] == "open_ended"]

    def _sum(lst): return sum(r["score"] for r in lst)
    def _max(lst): return sum(r["max_score"] for r in lst)
    def _pct(e, m): return round(100 * e / m, 1) if m else None

    total_e, total_m = _sum(scorable), _max(scorable)
    mc_e,    mc_m    = _sum(mc_results), _max(mc_results)
    oe_e,    oe_m    = _sum(oe_results), _max(oe_results)

    summary = {
        "model":            model_name,
        "exam":             exam_name,
        "rag":              use_rag,
        "total_score":      round(total_e, 2),
        "total_max":        total_m,
        "total_pct":        _pct(total_e, total_m),
        "mc_score":         round(mc_e, 2),
        "mc_max":           mc_m,
        "mc_pct":           _pct(mc_e, mc_m),
        "oe_score":         round(oe_e, 2),
        "oe_max":           oe_m,
        "oe_pct":           _pct(oe_e, oe_m),
        "questions_graded": len(scorable),
        "questions_total":  len(questions),
    }

    print(f"\n  Total : {total_e}/{total_m}  ({summary['total_pct']}%)")
    print(f"  MC    : {mc_e}/{mc_m}  ({summary['mc_pct']}%)")
    print(f"  OE    : {oe_e}/{oe_m}  ({summary['oe_pct']}%)")

    return {"summary": summary, "questions": results}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Evaluate a model on WISEflow exam JSON(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exam",     type=str, help="Single exam JSON file")
    g.add_argument("--exam-dir", type=str, help="Directory of exam JSONs (runs all *.json)")

    p.add_argument("--backend",    required=True, choices=["hf", "custom"])
    p.add_argument("--model-id",   type=str, help="HuggingFace model ID (hf backend)")
    p.add_argument("--lora-path",  type=str,
                   help="Root qwen_lora/ dir with adapter_config.json (hf backend)")
    p.add_argument("--checkpoint", type=str, help=".pt checkpoint file (custom backend)")
    p.add_argument("--model-name", type=str, help="Display name for result files")

    p.add_argument("--no-judge",       action="store_true",
                   help="Skip open-ended scoring entirely")
    p.add_argument("--judge-model-id", type=str,
                   help="Local HF judge model, e.g. Qwen/Qwen2.5-7B-Instruct")

    p.add_argument("--use-rag",        action="store_true",
                   help="Prepend retrieved lecture chunks to each question prompt")
    p.add_argument("--rag-index-dir",  type=str,
                   help="Override path to rag_index/ (default: workspace/rag/rag_index/)")
    p.add_argument("--rag-top-k",      type=int, default=3,
                   help="Number of chunks to retrieve per question (default: 3)")

    p.add_argument("--out-dir",        type=str, default="results/exam")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature",    type=float, default=0.7,
                   help="Sampling temperature (custom backend; hf uses greedy)")

    args = p.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    is_custom = args.backend == "custom"
    if is_custom:
        if not args.checkpoint:
            p.error("--checkpoint required for custom backend")
        model = CustomModel(
            args.checkpoint,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        model_name = args.model_name or Path(args.checkpoint).stem
    else:
        if not args.model_id:
            p.error("--model-id required for hf backend")
        model = HFModel(args.model_id, lora_path=args.lora_path,
                        max_new_tokens=args.max_new_tokens)
        if args.model_name:
            model_name = args.model_name
        elif args.lora_path:
            model_name = f"{args.model_id.split('/')[-1]}_lora"
        else:
            model_name = args.model_id.split("/")[-1]

    # ── Load judge ────────────────────────────────────────────────────────────
    judge_model = None
    if not args.no_judge:
        if args.judge_model_id:
            print(f"\nLoading judge: {args.judge_model_id}")
            judge_model = HFModel(args.judge_model_id, max_new_tokens=300)
        else:
            print("  Note: no --judge-model-id given; open-ended questions will not be scored.")

    # ── Load RAG retriever ────────────────────────────────────────────────────
    retriever = None
    if args.use_rag:
        # rag_retriever.py lives in the same eval/ directory
        eval_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(eval_dir))
        from rag_retriever import RAGRetriever
        retriever = RAGRetriever(
            index_dir=args.rag_index_dir,   # None → uses default workspace path
            top_k=args.rag_top_k,
        )

    # ── Collect exams ─────────────────────────────────────────────────────────
    if args.exam:
        exam_files = [Path(args.exam)]
    else:
        exam_files = sorted(Path(args.exam_dir).glob("*.json"))
        if not exam_files:
            print(f"No JSON files found in {args.exam_dir}")
            sys.exit(1)
        print(f"Found {len(exam_files)} exam(s) in {args.exam_dir}")

    # ── Run ───────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rag_suffix = "_rag" if args.use_rag else ""

    for exam_path in exam_files:
        output = evaluate_exam(
            exam_path=str(exam_path),
            model=model,
            model_name=model_name,
            is_custom=is_custom,
            judge_model=judge_model,
            retriever=retriever,
        )
        out_file = out_dir / f"{model_name}{rag_suffix}_{exam_path.stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_file}")

    # One-line summary table when multiple exams were run
    if len(exam_files) > 1:
        print(f"\n{'='*65}")
        print(f"Summary — {model_name}{rag_suffix}")
        print(f"{'Exam':<40} {'Total':>10}  {'MC':>8}  {'OE':>8}")
        print("-" * 65)
        for exam_path in exam_files:
            out_file = out_dir / f"{model_name}{rag_suffix}_{exam_path.stem}.json"
            if out_file.exists():
                with open(out_file) as f:
                    s = json.load(f)["summary"]
                print(f"  {exam_path.stem:<38} "
                      f"{s['total_score']}/{s['total_max']} ({s['total_pct']}%)  "
                      f"{s['mc_score']}/{s['mc_max']}  "
                      f"{s['oe_score']}/{s['oe_max']}")
        print("=" * 65)


if __name__ == "__main__":
    main()