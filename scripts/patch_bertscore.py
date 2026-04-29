"""
Patch missing BERTScore in eval/results/*/summary.json by recomputing
from per_example.jsonl.

Background:
-----------
`bert-score` 0.3.13 calls `tokenizer.build_inputs_with_special_tokens`,
which was removed from `transformers` 5.x.  We monkey-patch the method
back onto the relevant tokenizer base classes — it's a trivial 2-line
function that just prepends/appends the model's special tokens.

Idempotent: skips runs that already have a bertscore_f1 metric.
"""
import json
import sys
from pathlib import Path


def _patch_tokenizers():
    from transformers import (
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )

    def _build(self, token_ids_0, token_ids_1=None):
        cls = [self.cls_token_id] if self.cls_token_id is not None else []
        sep = [self.sep_token_id] if self.sep_token_id is not None else []
        if token_ids_1 is None:
            return cls + list(token_ids_0) + sep
        return cls + list(token_ids_0) + sep + list(token_ids_1) + sep

    for cls in (PreTrainedTokenizer, PreTrainedTokenizerFast):
        if not hasattr(cls, "build_inputs_with_special_tokens"):
            cls.build_inputs_with_special_tokens = _build  # type: ignore[attr-defined]


_patch_tokenizers()
from bert_score import score as bert_score_fn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"

if not RESULTS.exists():
    print(f"[patch] {RESULTS} not found")
    sys.exit(1)

for run_dir in sorted(RESULTS.iterdir()):
    summary_path = run_dir / "summary.json"
    per_path = run_dir / "per_example.jsonl"
    if not (summary_path.exists() and per_path.exists()):
        continue

    summary = json.loads(summary_path.read_text())
    # Recompute even if a bertscore_f1 is present — different runs in this
    # batch were scored with different fallback models earlier.  We force
    # all of them to roberta-large now so the numbers are comparable.
    existing = summary.get("metrics_mean", {}).get("bertscore_f1")
    if existing is not None and summary.get("bertscore_model") == "roberta-large":
        print(f"[patch] {run_dir.name}: bertscore_f1 already from roberta-large, skipping")
        continue

    preds, golds = [], []
    for line in per_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        preds.append(rec.get("pred", "") or "")
        golds.append(rec.get("gold", "") or "")
    if not preds:
        print(f"[patch] {run_dir.name}: no preds, skipping")
        continue

    print(f"[patch] {run_dir.name}: scoring {len(preds)} pairs with roberta-large")
    try:
        P, R, F = bert_score_fn(
            cands=preds,
            refs=golds,
            model_type="roberta-large",
            verbose=False,
            rescale_with_baseline=False,
            batch_size=32,
        )
    except Exception as e:
        print(f"[patch] {run_dir.name}: BERTScore failed: {e}")
        continue

    p, r, f = float(P.mean()), float(R.mean()), float(F.mean())
    summary.setdefault("metrics_mean", {})
    summary["metrics_mean"]["bertscore_p"] = p
    summary["metrics_mean"]["bertscore_r"] = r
    summary["metrics_mean"]["bertscore_f1"] = f
    summary.setdefault("metrics_count", {})
    summary["metrics_count"]["bertscore_f1"] = len(preds)
    summary["bertscore_model"] = "roberta-large"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[patch] {run_dir.name}: BERTScore P={p:.4f} R={r:.4f} F1={f:.4f}")

print("[patch] done")
