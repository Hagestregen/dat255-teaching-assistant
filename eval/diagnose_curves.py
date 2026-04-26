"""
diagnose_curves.py  —  Decide whether a training run is under- or overfit
=========================================================================
Reads the JSON file written by train.py's `--log-curves` flag and applies
the diagnostic gates from the SLM-improvement plan (Phase 7):

    UNDERFITTING  →  train loss still decreasing at the last logged step,
                     train and val loss both high and roughly equal.
                     Action: more data, longer training, or a LARGER model.

    OVERFITTING   →  val loss flattens / rises while train keeps falling
                     (large positive train→val gap).
                     Action: more regularization or more data — NOT a
                     larger model.

    SWEET SPOT    →  both decreasing slowly, stable small gap.
                     Action: ship and move on.

Usage
-----
    python eval/diagnose_curves.py runs/curves_baseline.json

Optionally pass two paths to compare a baseline vs. a variant:

    python eval/diagnose_curves.py runs/curves_baseline.json \\
                                   runs/curves_variant.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def load_curves(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _last_n(records: list, n: int) -> list:
    return records[-n:] if len(records) >= n else records


def _slope(records: list, key: str) -> Optional[float]:
    """Approximate slope of `key` per step over the last ~25% of records."""
    if len(records) < 4:
        return None
    tail = _last_n(records, max(4, len(records) // 4))
    xs = [r["step"] for r in tail]
    ys = [r[key]    for r in tail]
    n  = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    return num / den if den else None


def diagnose(curves: dict) -> dict:
    train = curves.get("train", [])
    val   = curves.get("val", [])

    if len(train) < 2 or len(val) < 2:
        return {"verdict": "insufficient_data", "reason": "Need at least 2 train and 2 val records."}

    final_train_loss = train[-1]["loss"]
    final_val_loss   = val[-1]["val_loss"]
    best_val_loss    = min(r["val_loss"] for r in val)
    val_gap          = final_train_loss - final_val_loss
    train_slope      = _slope(train, "loss")
    val_slope        = _slope(val,   "val_loss")

    # Heuristics tuned for short runs (<10k steps).  A negative slope means
    # the curve is still going down; magnitude is per-step.
    train_still_falling = train_slope is not None and train_slope < -1e-5
    val_flat_or_rising  = val_slope   is None or val_slope >= -1e-6
    val_rose_from_best  = final_val_loss > best_val_loss + 0.02

    if val_rose_from_best or (val_flat_or_rising and val_gap < -0.05):
        verdict   = "overfitting"
        rec       = ("Val rose above its best while train kept falling. "
                     "Add data or regularization (label smoothing, dropout, EMA, stochastic depth). "
                     "Do NOT scale the model up.")
    elif train_still_falling and not val_flat_or_rising:
        verdict   = "still_learning"
        rec       = ("Both losses are still decreasing.  Continue training "
                     "before deciding on model size.")
    elif train_still_falling and val_flat_or_rising:
        verdict   = "underfitting"
        rec       = ("Train still falling, val plateau, gap small. "
                     "Consider scaling the model up (next preset) or generating more data.")
    elif not train_still_falling and not val_flat_or_rising:
        verdict   = "sweet_spot"
        rec       = "Both curves slowing down with a stable small gap.  Ship it."
    else:
        verdict   = "noisy"
        rec       = "Curves are inconclusive; consider a longer run with more frequent val evals."

    return {
        "verdict":            verdict,
        "recommendation":     rec,
        "final_train_loss":   final_train_loss,
        "final_val_loss":     final_val_loss,
        "best_val_loss":      best_val_loss,
        "train_slope":        train_slope,
        "val_slope":          val_slope,
        "train_val_gap":      val_gap,
        "n_train_records":    len(train),
        "n_val_records":      len(val),
    }


def _print_diagnosis(label: str, d: dict):
    print(f"\n=== {label} ===")
    if d["verdict"] == "insufficient_data":
        print("  Not enough data to diagnose.")
        return
    print(f"  verdict          : {d['verdict']}")
    print(f"  final train loss : {d['final_train_loss']:.4f}")
    print(f"  final val   loss : {d['final_val_loss']:.4f}")
    print(f"  best  val   loss : {d['best_val_loss']:.4f}")
    print(f"  train slope (Δ/step) : {d['train_slope']}")
    print(f"  val   slope (Δ/step) : {d['val_slope']}")
    print(f"  recommendation   : {d['recommendation']}")


def _next_size(current: str) -> str:
    progression = ["25M", "50M", "125M"]
    if current not in progression or current == progression[-1]:
        return current
    return progression[progression.index(current) + 1]


def main():
    p = argparse.ArgumentParser(description="Diagnose training curves (Phase 7).")
    p.add_argument("paths", nargs="+", help="One or more curves JSON files")
    p.add_argument("--current-size", default=None,
                   help="If known, which model size produced the curves (25M, 50M, 125M)")
    args = p.parse_args()

    diagnoses = []
    for path in args.paths:
        path = Path(path)
        if not path.exists():
            print(f"  [skip] {path} does not exist")
            continue
        d = diagnose(load_curves(path))
        diagnoses.append((path, d))
        _print_diagnosis(str(path), d)

    if args.current_size and diagnoses:
        last = diagnoses[-1][1]
        if last["verdict"] == "underfitting":
            nxt = _next_size(args.current_size)
            if nxt != args.current_size:
                print(f"\nPhase 7 recommendation: scale up {args.current_size} → {nxt}")
            else:
                print("\nAlready at the largest preset; consider more data instead.")
        elif last["verdict"] == "overfitting":
            print(f"\nPhase 7 recommendation: keep {args.current_size}; add data / regularization.")
        elif last["verdict"] == "sweet_spot":
            print(f"\nPhase 7 recommendation: ship at {args.current_size}.")


if __name__ == "__main__":
    main()
