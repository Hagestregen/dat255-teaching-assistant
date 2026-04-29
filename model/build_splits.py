"""
build_splits.py  —  Freeze a stratified train/val/test split to disk.
=====================================================================

Why this script exists
----------------------
`dataset.build_datasets()` used to re-split the data on every training run.
That meant two runs trained on slightly different examples and were therefore
not strictly comparable.  Worse, when `data_generation.py` ran in the
background it grew `train.jsonl` between runs, silently shifting the split.

This script splits the data ONCE and writes three JSONL files plus a
manifest.  All subsequent `train.py` and `run_eval.py` invocations read
those files directly so every comparison in the report is over identical
examples.

What it does
------------
1. Loads HuggingFace QA datasets (deduplicated by question+answer text).
2. Loads locally generated examples (from data_generation.py).
3. For locally generated examples, groups by `chunk_hash` so the four
   examples produced from the same chunk all land in the same split — this
   prevents content leakage from train into val/test.
4. Stratifies each split by `(source, task_type)` so the per-task and
   per-source distribution is roughly preserved.
5. Writes `model/data/splits/{train,val,test}.jsonl` plus a `manifest.json`
   recording counts, fractions, seed, source hashes and per-split task
   breakdown.

Usage
-----
    python model/build_splits.py \\
        --synthetic-jsonl model/data/train.jsonl \\
        --output-dir model/data/splits \\
        --val-frac 0.05 --test-frac 0.10 --seed 42

    # Inspect what was produced:
    python model/build_splits.py --inspect model/data/splits

Notes
-----
- HuggingFace examples have no `chunk_hash`, so they are split per-example.
- We rely on the existing `normalize_example()` from dataset.py so the schema
  is consistent with what training code already expects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Reuse the loaders + normaliser from dataset.py so we never disagree about schema.
from dataset import (
    load_huggingface_datasets,
    load_local_generated_data,
    normalize_example,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _file_sha256(path: Path) -> str:
    """Quick fingerprint for a source file so the manifest can record it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stratified_take(
    examples: List[Dict],
    val_frac: float,
    test_frac: float,
    rng: random.Random,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split a flat list of examples into (train, val, test) such that the
    proportion of every (source, type) bucket is preserved within rounding.
    """
    # Bucket examples by stratification key.
    buckets: dict[tuple[str, str], list[Dict]] = defaultdict(list)
    for ex in examples:
        key = (ex.get("source", "unknown"), ex.get("type", "explanation"))
        buckets[key].append(ex)

    train, val, test = [], [], []
    for key, items in buckets.items():
        rng.shuffle(items)
        n        = len(items)
        n_test   = int(round(n * test_frac))
        n_val    = int(round(n * val_frac))
        # Edge case: tiny buckets — guarantee at least one sample lands in
        # train so the bucket isn't entirely held out.
        if n - n_test - n_val < 1 and n >= 3:
            n_test = max(1, n_test)
            n_val  = max(1, n_val)
            if n - n_test - n_val < 1:
                n_val = max(0, n - n_test - 1)
        test_  = items[:n_test]
        val_   = items[n_test: n_test + n_val]
        train_ = items[n_test + n_val:]
        train += train_
        val   += val_
        test  += test_
    return train, val, test


def _split_course_by_chunk(
    examples: List[Dict],
    val_frac: float,
    test_frac: float,
    rng: random.Random,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    For locally-generated examples, the leakage unit is `chunk_hash`: the
    same lecture chunk can produce up to four examples (explanation, quiz,
    review, flashcard) and all of them must end up in the same split or
    the test set is contaminated.

    We therefore split CHUNKS by val_frac/test_frac, then flatten back to
    examples.  Within each chunk-level split we still try to preserve the
    task-type proportions by stratifying on type AFTER the chunk-level cut.
    """
    by_chunk: dict[str, list[Dict]] = defaultdict(list)
    no_hash:  list[Dict]            = []
    for ex in examples:
        h = ex.get("chunk_hash")
        if h:
            by_chunk[h].append(ex)
        else:
            no_hash.append(ex)

    chunk_ids = list(by_chunk.keys())
    rng.shuffle(chunk_ids)

    n        = len(chunk_ids)
    n_test   = int(round(n * test_frac))
    n_val    = int(round(n * val_frac))
    test_ids  = set(chunk_ids[:n_test])
    val_ids   = set(chunk_ids[n_test: n_test + n_val])
    train_ids = set(chunk_ids[n_test + n_val:])

    train = [ex for cid in train_ids for ex in by_chunk[cid]]
    val   = [ex for cid in val_ids   for ex in by_chunk[cid]]
    test  = [ex for cid in test_ids  for ex in by_chunk[cid]]

    # Examples without chunk_hash (shouldn't happen for current pipeline,
    # but defensive): fall back to plain stratified split so we don't drop them.
    if no_hash:
        t2, v2, te2 = _stratified_take(no_hash, val_frac, test_frac, rng)
        train += t2; val += v2; test += te2

    return train, val, test


def _task_breakdown(examples: List[Dict]) -> Dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex.get("type", "explanation")] += 1
    return dict(sorted(counts.items()))


def _source_breakdown(examples: List[Dict]) -> Dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex.get("source", "unknown")] += 1
    return dict(sorted(counts.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Main split build
# ─────────────────────────────────────────────────────────────────────────────

def build_and_persist_splits(
    synthetic_jsonl_path: str,
    output_dir: str,
    val_frac: float = 0.05,
    test_frac: float = 0.10,
    seed: int = 42,
    skip_general: bool = False,
) -> Dict:
    """
    Run the full pipeline: load → normalise → split → write JSONL + manifest.
    Returns the manifest dict (also written to disk).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # ── 1. Load and normalise sources ─────────────────────────────────────────
    if skip_general:
        general: List[Dict] = []
        print("Skipping HuggingFace QA datasets (--skip-general).")
    else:
        general = load_huggingface_datasets()

    course = load_local_generated_data(synthetic_jsonl_path) if synthetic_jsonl_path else []

    # ── 2. Deduplicate ──────────────────────────────────────────────────────
    # HuggingFace datasets have a non-trivial duplicate rate (same question
    # appears in both prsdm and win-wang).  Dedup by (question, answer) text.
    seen_pairs: set[tuple[str, str]] = set()
    deduped_general: List[Dict] = []
    for ex in general:
        key = (ex["question"].strip().lower(), ex["answer"].strip().lower())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped_general.append(ex)
    n_dups = len(general) - len(deduped_general)
    if n_dups:
        print(f"Deduped {n_dups} duplicate HuggingFace examples.")

    # ── 3. Split each source ─────────────────────────────────────────────────
    g_train, g_val, g_test = _stratified_take(deduped_general, val_frac, test_frac, rng)
    c_train, c_val, c_test = _split_course_by_chunk(course,    val_frac, test_frac, rng)

    train = g_train + c_train
    val   = g_val   + c_val
    test  = g_test  + c_test

    rng.shuffle(train)
    # val/test are NOT shuffled so they remain stable across runs that
    # extend the dataset later (new examples just appear at the end).

    # ── 4. Write JSONL files ─────────────────────────────────────────────────
    def _write_jsonl(examples: List[Dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(examples):5d} examples → {path}")

    _write_jsonl(train, out / "train.jsonl")
    _write_jsonl(val,   out / "val.jsonl")
    _write_jsonl(test,  out / "test.jsonl")

    # ── 5. Manifest ──────────────────────────────────────────────────────────
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "skip_general": skip_general,
        "synthetic_jsonl_sha256": (
            _file_sha256(Path(synthetic_jsonl_path))
            if synthetic_jsonl_path and Path(synthetic_jsonl_path).exists()
            else None
        ),
        "counts": {
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
        },
        "tasks": {
            "train": _task_breakdown(train),
            "val":   _task_breakdown(val),
            "test":  _task_breakdown(test),
        },
        "sources": {
            "train": _source_breakdown(train),
            "val":   _source_breakdown(val),
            "test":  _source_breakdown(test),
        },
        "course_chunks": {
            "train_unique": len({e["chunk_hash"] for e in c_train if e.get("chunk_hash")}),
            "val_unique":   len({e["chunk_hash"] for e in c_val   if e.get("chunk_hash")}),
            "test_unique":  len({e["chunk_hash"] for e in c_test  if e.get("chunk_hash")}),
        },
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote manifest             → {manifest_path}")

    # ── 6. Sanity: no chunk_hash should appear in two splits ────────────────
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        a_ids = {e["chunk_hash"] for e in (train if a == "train" else val if a == "val" else test) if e.get("chunk_hash")}
        b_ids = {e["chunk_hash"] for e in (train if b == "train" else val if b == "val" else test) if e.get("chunk_hash")}
        overlap = a_ids & b_ids
        if overlap:
            raise RuntimeError(f"chunk_hash leakage between {a} and {b}: {len(overlap)} chunks "
                               f"appear in both splits — refusing to ship contaminated splits.")
    print("Leakage check: no chunk_hash appears in more than one split.")

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Inspect existing splits
# ─────────────────────────────────────────────────────────────────────────────

def inspect(splits_dir: str) -> None:
    p = Path(splits_dir)
    manifest = json.loads((p / "manifest.json").read_text())
    print(json.dumps(manifest, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen train/val/test split")
    parser.add_argument("--synthetic-jsonl", default="model/data/train.jsonl",
                        help="Path to locally generated examples")
    parser.add_argument("--output-dir",      default="model/data/splits",
                        help="Where to write {train,val,test}.jsonl + manifest.json")
    parser.add_argument("--val-frac",  type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--skip-general", action="store_true",
                        help="Skip the HuggingFace QA datasets (course-only splits)")
    parser.add_argument("--inspect", metavar="DIR",
                        help="Print the manifest of an existing splits directory and exit")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    build_and_persist_splits(
        synthetic_jsonl_path = args.synthetic_jsonl,
        output_dir           = args.output_dir,
        val_frac             = args.val_frac,
        test_frac            = args.test_frac,
        seed                 = args.seed,
        skip_general         = args.skip_general,
    )


if __name__ == "__main__":
    main()
