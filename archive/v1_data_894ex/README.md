# v1 dataset baseline runs (archived)

These are the Phase A and Phase B runs from **2026-04-28**, trained against
the v1 synthetic dataset (`model/data/train.jsonl` at 894 lines).

They are preserved here as a baseline reference for the project report's
appendix.  All headline numbers in the main report come from the v2 runs,
which use the larger generated dataset and the frozen stratified splits.

## Contents

| Path | Description |
|---|---|
| `checkpoints/` | Best checkpoints per run (`best.pt`, `best_ema.pt`, `final.pt`) |
| `runs/` | Training curves JSON + raw stdout logs |
| `eval_results/` | Per-example outputs and summary metrics from Phase C eval |

## Run summary (best EMA val_loss)

| Run | Best EMA val | Notes |
|---|---|---|
| `scratch_125m` | 5.56 | A1 — 125M from scratch, no warm start |
| `gpt2_warm_data10` | 3.65 | A2 — 10 % of v1 train (~85 ex.) |
| `gpt2_warm_data25` | 3.38 | A2 — 25 % of v1 train (~212 ex.) |
| `gpt2_warm_data50` | 3.14 | A2 — 50 % of v1 train (~424 ex.) |
| `gpt2_warm_v2` | 2.95 | reference — full v1 data, rag-prob 0.5 |
| `gpt2_warm_rag00` | 2.94 | B1 — same recipe with rag-prob 0.0 |
| `run_25m_v2`, `run_25m_v3` | 5.46 / 5.77 | early experiments before warm-start adoption |

## Why these aren't in the main results

After these runs finished we re-generated the synthetic dataset (with the
4-task-type schema and additional course chunks), grew it from 894 → ~3000
examples, and added stratified train/val/test splits frozen on disk.  The
main report numbers come from training on those frozen splits so every run
trains and is evaluated on identical examples.
