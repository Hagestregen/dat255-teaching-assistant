# Experiment tracking — what to use for the project report

## Short answer

**Don't add MLflow.** This project already writes everything you need as JSON.
Use the local registry script for the report; if you want richer live plots,
flip on the W&B hook that's already wired into [model/train.py](../model/train.py).

## What we already have (zero new dependencies)

| Artifact | Where it's written | What's in it |
| --- | --- | --- |
| Per-step train metrics | `runs/curves_<NAME>.json` (train.py `--log-curves`) | step, loss, ppl, lr, grad_norm, tokens_per_sec |
| Per-val metrics       | `runs/curves_<NAME>.json`                     | step, val_loss, val_ppl, ema_val_loss, ema_val_ppl |
| Eval predictions      | `eval/results/<NAME>/per_example.jsonl`       | question, gold, pred, scores |
| Eval aggregate        | `eval/results/<NAME>/summary.json`            | metric means, n_examples, with_context, judge model |
| Curve verdict         | `python eval/diagnose_curves.py runs/curves_<NAME>.json` | overfitting / underfitting / sweet-spot / still-learning |
| Comparison table      | `python eval/compare_runs.py --out docs/experiments.md` | one Markdown table per run, registry-style |

That's enough source data for any plot a report would need.

## When to add a real tracker

| Situation | Recommended tool | Effort |
| --- | --- | --- |
| Just need a writeup with numbers and a few plots | **Local JSON + `compare_runs.py`** | 0 — already done |
| Want shareable, browser-viewable charts during training | **Weights & Biases (`--use-wandb`)** | 5 min: `pip install wandb && wandb login`, add the flag |
| Need scrollable per-step charts but no cloud account | **TensorBoard** | 30 min: write a small SummaryWriter hook in train.py |
| Multi-author, multi-experiment, model registry, prod | **MLflow** | Hours: server, schema, integration — overkill here |

## How to use what's wired up

### Naming convention

Every training command should pass `--log-curves runs/curves_<NAME>.json` and
`--out-dir checkpoints/<NAME>` with the *same* `<NAME>`.  Then matching eval
result dirs become `eval/results/<NAME>` (or `eval/results/<NAME>_<variant>`
if you want to compare e.g. live vs. EMA weights).  `compare_runs.py` walks
all `runs/curves_*.json` and groups by that name.

### Standard report-ready table

```bash
python eval/compare_runs.py --out docs/experiments.md
```

regenerates [docs/experiments.md](experiments.md) from disk; commit that file
or paste its contents into the report.

### Live charts via W&B (already in the code)

[model/train.py](../model/train.py) already has the W&B hooks:

```bash
pip install wandb
wandb login            # paste your key from wandb.ai/settings
python model/train.py --use-wandb ...other-flags
```

Project name is set inside `train.py` (`project="dat255-teaching-assistant"`).
You'll see live train_loss / val_loss / lr / grad_norm / tokens_per_sec
curves at `https://wandb.ai/<your-user>/dat255-teaching-assistant`.

## What to write in the report

For each experiment, the report can cite three things — and all three are
already on disk after a run finishes:

1. **Setup**: the resolved configs printed at the top of `runs/run_<NAME>.log`
   (TrainingConfig + TransformerConfig).
2. **Curves**: `runs/curves_<NAME>.json` plotted with matplotlib, or a screenshot
   of `python eval/diagnose_curves.py runs/curves_<NAME>.json`.
3. **Held-out scores**: the row from `python eval/compare_runs.py`.

## Minimal, repeatable workflow

```bash
NAME=gpt2_warm_v2
python model/train.py --preset 3090 --model-size 125M \
    --resume checkpoints/gpt2_init.pt \
    --log-curves runs/curves_$NAME.json \
    --out-dir checkpoints/$NAME \
    --synthetic-jsonl model/data/train.jsonl

python eval/diagnose_curves.py runs/curves_$NAME.json --current-size 125M

python eval/run_eval.py \
    --checkpoint checkpoints/$NAME/best.pt \
    --output-dir eval/results/${NAME}_live \
    --synthetic-jsonl model/data/train.jsonl

python eval/run_eval.py \
    --checkpoint checkpoints/$NAME/best_ema.pt \
    --output-dir eval/results/${NAME}_ema \
    --synthetic-jsonl model/data/train.jsonl --use-ema

python eval/compare_runs.py --out docs/experiments.md
```

Now `docs/experiments.md` is the canonical "what worked, what didn't" table
for the report.
