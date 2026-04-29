"""
run_eval.py  —  End-to-end evaluation harness for a trained checkpoint
======================================================================
Loads a saved checkpoint, generates predictions on a fixed held-out test
split, and computes per-example + aggregate metrics.

Design choices:
  - The test split is persisted to `eval/test_split.json` after the FIRST
    run.  Subsequent runs reload from this file so that all checkpoints
    are evaluated on the same examples (otherwise re-running data
    generation would silently shift the split).
  - Per-example records (question, gold, prediction, scores) are written
    to a JSONL file so failure cases can be inspected manually.
  - Aggregate stats (mean per metric, counts) are written to summary.json.
  - LLM-judge is optional and disabled by default to keep the harness fast.
    Pass --judge to enable it.

Usage
-----
    # First-ever evaluation: builds and saves the test split
    python eval/run_eval.py --checkpoint model/checkpoints/baseline.pt \
        --output-dir eval/results/baseline

    # With retrieved-context formatting
    python eval/run_eval.py --checkpoint model/checkpoints/variant_rag.pt \
        --output-dir eval/results/variant_rag_with_context \
        --with-context

    # With LLM-judge
    python eval/run_eval.py --checkpoint model/checkpoints/baseline.pt \
        --output-dir eval/results/baseline --judge \
        --judge-model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Match the existing project layout: model/ uses module-style imports
# internally (e.g. `from transformer import ...`).  We add the model/ and
# eval/ directories to sys.path so we can do the same here.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "model"))
sys.path.insert(0, str(_ROOT / "eval"))

from dataset import Tokenizer, build_datasets                            # noqa: E402
from train   import load_checkpoint                                      # noqa: E402
from metrics import (                                                    # noqa: E402
    exact_match, token_f1, rouge_l, llm_judge, aggregate_scores,
    meteor_score_safe, bertscore_corpus,
)


TEST_SPLIT_PATH = _ROOT / "eval" / "test_split.json"


# ─────────────────────────────────────────────────────────────────────────────
# Test-split persistence
# ─────────────────────────────────────────────────────────────────────────────

FROZEN_SPLITS_DEFAULT = _ROOT / "model" / "data" / "splits"


def get_or_build_test_split(
    chunks_json_path: str,
    synthetic_jsonl_path: str,
    tokenizer: Tokenizer,
    max_length: int,
    splits_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Return the canonical test set as a list of {question, answer, source,
    context?, type?} dicts.

    Priority:
      1. The frozen splits directory (`model/data/splits/test.jsonl`) if it
         exists.  This is the preferred source — every training run reads
         the matching `train.jsonl` so all comparisons are over identical
         examples.
      2. The legacy `eval/test_split.json` cache from earlier runs.
      3. Build a new split via `build_datasets()` (legacy on-the-fly path).
    """
    splits_dir = Path(splits_dir or FROZEN_SPLITS_DEFAULT)
    frozen_path = splits_dir / "test.jsonl"
    if frozen_path.exists():
        examples: list[dict] = []
        with open(frozen_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        print(f"Loaded frozen test split: {len(examples)} examples from {frozen_path}")
        return examples

    if TEST_SPLIT_PATH.exists():
        with open(TEST_SPLIT_PATH, encoding="utf-8") as f:
            examples = json.load(f)
        print(f"Loaded legacy test split: {len(examples)} examples from {TEST_SPLIT_PATH}")
        return examples

    print("Building test split for the first time...")
    _, _, test_ds = build_datasets(
        synthetic_jsonl_path=synthetic_jsonl_path,
        chunks_json_path=chunks_json_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    examples: list[dict] = []
    for raw in test_ds.raw_examples if hasattr(test_ds, "raw_examples") else []:
        examples.append({
            "question": raw["question"],
            "answer":   raw["answer"],
            "source":   raw.get("source", "unknown"),
            "context":  raw.get("context", ""),
        })

    if not examples:
        # Backwards-compatible fallback: extract from tokenized dataset.
        # `QADataset` stores raw normalized examples internally as well; if not
        # exposed, we ask the user to regenerate.
        raise RuntimeError(
            "Could not extract raw examples from the test dataset. "
            "Update QADataset to expose raw_examples (Phase 3) and rerun."
        )

    TEST_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Persisted test split ({len(examples)} examples) → {TEST_SPLIT_PATH}")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_prompt(example: dict, with_context: bool) -> str:
    """
    Format an example as the prompt the model expects.
    Mirrors format_for_training with the answer/EOT removed.
    """
    q = example["question"]
    ctx = (example.get("context") or "").strip() if with_context else ""
    if ctx:
        return f"Context: {ctx}\n\nQuestion: {q}\nAnswer:"
    return f"Question: {q}\nAnswer:"


@torch.no_grad()
def generate_answer(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    """Generate an answer string from a prompt."""
    ids  = tokenizer.encode(prompt)
    x    = torch.tensor([ids], dtype=torch.long, device=device)
    eot  = tokenizer.eot_id
    out  = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_token=eot,
    )
    new_ids = out[0, x.size(1):].tolist()
    # Trim the trailing EOT if present
    if new_ids and new_ids[-1] == eot:
        new_ids = new_ids[:-1]
    return tokenizer.decode(new_ids).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main eval loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    chunks_json_path: str,
    synthetic_jsonl_path: str,
    max_length: int,
    max_examples: int,
    with_context: bool,
    use_judge: bool,
    judge_model: str,
    max_new_tokens: int,
    temperature: float,
    use_ema: bool = False,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model, _opt_state, step, ckpt_metrics = load_checkpoint(checkpoint_path, device)
    model.eval()

    # Optionally swap EMA shadow weights into the model for evaluation.
    # The EMA shadow tends to give a small but consistent generalization
    # bump; on this project's data it has been worth ~0.5–1.0 nats of val
    # loss vs. the live weights at the same step.
    if use_ema:
        ckpt = torch.load(checkpoint_path, map_location=device)
        ema_state = ckpt.get("ema_state")
        if not ema_state:
            raise RuntimeError(
                f"--use-ema requested but {checkpoint_path} has no ema_state. "
                f"Re-train with --ema-decay > 0 to populate it."
            )
        loaded = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in ema_state:
                    p.data.copy_(ema_state[name].to(device))
                    loaded += 1
        print(f"Loaded {loaded} EMA shadow tensors into the model for evaluation.")
        del ckpt

    tokenizer = Tokenizer()

    examples = get_or_build_test_split(
        chunks_json_path=chunks_json_path,
        synthetic_jsonl_path=synthetic_jsonl_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    if max_examples and max_examples < len(examples):
        examples = examples[:max_examples]
        print(f"Truncating to first {max_examples} examples")

    per_example_path = out_dir / "per_example.jsonl"
    summary_path     = out_dir / "summary.json"

    print(f"\nGenerating predictions for {len(examples)} examples "
          f"(with_context={with_context}, judge={use_judge})...")

    records: list[dict] = []
    t0 = time.time()
    with open(per_example_path, "w", encoding="utf-8") as f_out:
        for i, ex in enumerate(examples):
            prompt = make_prompt(ex, with_context=with_context)
            try:
                pred = generate_answer(
                    model, tokenizer, prompt, device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"  [{i}] generation error: {e}")
                pred = ""

            scores: dict = {
                "exact_match": exact_match(pred, ex["answer"]),
                "token_f1":    token_f1(pred, ex["answer"]),
                "rouge_l":     rouge_l(pred, ex["answer"]),
            }
            # METEOR is cheap and runs inline; BERTScore is batched after
            # the loop so we don't reload its 110M-param model per example.
            meteor = meteor_score_safe(pred, ex["answer"])
            if meteor is not None:
                scores["meteor"] = meteor

            judge_out: dict | None = None
            if use_judge:
                judge_out = llm_judge(
                    question=ex["question"],
                    pred=pred,
                    gold=ex["answer"],
                    model_id=judge_model,
                )
                if judge_out is not None:
                    for k in ("correctness", "completeness", "pedagogical_clarity"):
                        scores[f"judge_{k}"] = judge_out[k]

            record = {
                "i":        i,
                "question": ex["question"],
                "gold":     ex["answer"],
                "context":  ex.get("context", "") if with_context else "",
                "pred":     pred,
                "scores":   scores,
                "judge":    judge_out,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

            if (i + 1) % 10 == 0 or i + 1 == len(examples):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"  [{i+1}/{len(examples)}]  {rate:.2f} ex/s")

    # ── Corpus-level BERTScore (batched, runs once over all preds) ──────────
    print("\nComputing BERTScore over all predictions...")
    bs = bertscore_corpus(
        preds=[r["pred"] for r in records],
        golds=[r["gold"] for r in records],
    )
    if bs is not None:
        for r, scores in zip(records, bs):
            r["scores"].update(scores)
        # Re-write the per-example file so it includes BERTScore.  Cheap:
        # only happens once per eval call.
        with open(per_example_path, "w", encoding="utf-8") as f_out:
            for r in records:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    agg = aggregate_scores(records)
    summary = {
        "checkpoint":        str(checkpoint_path),
        "step":              step,
        "checkpoint_metrics": ckpt_metrics,
        "n_examples":        agg["n"],
        "with_context":      with_context,
        "judge_used":        use_judge,
        "judge_model":       judge_model if use_judge else None,
        "metrics_mean":      agg["means"],
        "metrics_count":     agg["counts"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=" * 5 + " summary " + "=" * 5)
    for k, v in summary["metrics_mean"].items():
        print(f"  {k:25s}  {v:.4f}")
    print(f"\nPer-example records: {per_example_path}")
    print(f"Summary:             {summary_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate a trained Teaching Assistant checkpoint.")
    p.add_argument("--checkpoint",      required=True, help="Path to .pt checkpoint")
    p.add_argument("--output-dir",      required=True, help="Where to write per-example + summary")
    p.add_argument("--chunks",          default="rag/rag_index/chunks.json",
                   help="Path to chunks.json (used only if test split not yet persisted)")
    p.add_argument("--synthetic-jsonl", default="model/data/train.jsonl",
                   help="Path to synthetic data jsonl (used only on first run)")
    p.add_argument("--max-length",      type=int,   default=512)
    p.add_argument("--max-examples",    type=int,   default=200,
                   help="Max test examples to evaluate (0 = all)")
    p.add_argument("--with-context",    action="store_true",
                   help="Format prompts with retrieved Context (RAG-style)")
    p.add_argument("--judge",           action="store_true",
                   help="Run LLM-as-judge (slow, requires GPU + Qwen download)")
    p.add_argument("--judge-model",     default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-new-tokens",  type=int,   default=200)
    p.add_argument("--temperature",     type=float, default=0.7)
    p.add_argument("--use-ema",         action="store_true",
                   help="Use the EMA shadow weights stored in the checkpoint")
    args = p.parse_args()

    evaluate_checkpoint(
        checkpoint_path      = args.checkpoint,
        output_dir           = args.output_dir,
        chunks_json_path     = args.chunks,
        synthetic_jsonl_path = args.synthetic_jsonl,
        max_length           = args.max_length,
        max_examples         = args.max_examples if args.max_examples > 0 else 0,
        with_context         = args.with_context,
        use_judge            = args.judge,
        judge_model          = args.judge_model,
        max_new_tokens       = args.max_new_tokens,
        temperature          = args.temperature,
        use_ema              = args.use_ema,
    )


if __name__ == "__main__":
    main()
