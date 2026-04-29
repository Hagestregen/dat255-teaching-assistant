"""
Compute a semantic-similarity metric for every Phase-C eval result so the
report has a uniform "embedding-based" column (the original BERTScore call
inside run_eval.py hit a bert-score / transformers v5 API mismatch on most
runs).

We use sentence-transformers/all-MiniLM-L6-v2 — small (22M params), fast,
and produces stable cosine similarities in [-1, 1] which we report as the
mean across the test set.

Idempotent: skips runs whose summary.json already has a "semantic_sim_mean".
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"

if not RESULTS.exists():
    print(f"[patch] {RESULTS} not found")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[patch] loading sentence encoder on {device}...")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

for run_dir in sorted(RESULTS.iterdir()):
    summary_path = run_dir / "summary.json"
    per_path = run_dir / "per_example.jsonl"
    if not (summary_path.exists() and per_path.exists()):
        continue

    summary = json.loads(summary_path.read_text())
    if summary.get("metrics_mean", {}).get("semantic_sim_mean") is not None:
        print(f"[patch] {run_dir.name}: semantic_sim_mean present, skipping")
        continue

    preds, golds = [], []
    for line in per_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        preds.append((rec.get("pred", "") or "").strip() or "<empty>")
        golds.append((rec.get("gold", "") or "").strip() or "<empty>")
    if not preds:
        continue

    print(f"[patch] {run_dir.name}: encoding {len(preds)} pred/gold pairs...")
    pred_emb = encoder.encode(preds, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
    gold_emb = encoder.encode(golds, batch_size=64, convert_to_tensor=True, show_progress_bar=False)
    sims = cos_sim(pred_emb, gold_emb).diag().cpu().numpy()
    mean_sim = float(np.mean(sims))
    median_sim = float(np.median(sims))

    summary.setdefault("metrics_mean", {})
    summary["metrics_mean"]["semantic_sim_mean"]   = mean_sim
    summary["metrics_mean"]["semantic_sim_median"] = median_sim
    summary.setdefault("metrics_count", {})
    summary["metrics_count"]["semantic_sim_mean"] = len(preds)
    summary["semantic_sim_model"] = "sentence-transformers/all-MiniLM-L6-v2"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[patch] {run_dir.name}: mean cos-sim {mean_sim:.4f} | median {median_sim:.4f}")

print("[patch] done")
