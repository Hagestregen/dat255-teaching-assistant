#!/usr/bin/env bash
# v5: GPT-2 small warm start, this time with:
#   - best_ema.pt tracking (added to train.py after v4 to capture the EMA peak)
#   - W&B live logging streamed to the dat255-teaching-assistant project
#   - all caches + the wandb directory pinned to /mnt/obelix_serving
#
# Override RUN_NAME to make a fresh run; everything (curves JSON, ckpt dir,
# W&B run name) is keyed off that single variable so eval/compare_runs.py
# groups them automatically.
set -u

cd "$(dirname "$0")/.."

RUN_NAME=${RUN_NAME:-gpt2_warm_v2}

export TMPDIR=/mnt/obelix_serving/tmp
export PIP_CACHE_DIR=/mnt/obelix_serving/pip-cache
export HF_HOME=/mnt/obelix_serving/hf_cache
export TRANSFORMERS_CACHE=/mnt/obelix_serving/hf_cache
export HF_DATASETS_CACHE=/mnt/obelix_serving/hf_cache
export TIKTOKEN_CACHE_DIR=/mnt/obelix_serving/hf_cache/tiktoken
export TORCHINDUCTOR_CACHE_DIR=/mnt/obelix_serving/tmp/torchinductor

# W&B: the team is already logged in via `wandb login`.  We pin the run
# directory to obelix_serving so wandb/ doesn't bloat the system disk, and
# we name the run identically to the curves JSON file so the UI, the
# filesystem, and docs/experiments.md all agree.
export WANDB_DIR=/mnt/obelix_serving/wandb_runs
export WANDB_NAME="$RUN_NAME"
export WANDB_RUN_GROUP=gpt2_warm

mkdir -p "$TMPDIR" "$TIKTOKEN_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_DIR"

source .venv/bin/activate
mkdir -p runs "checkpoints/$RUN_NAME"

INIT_PATH=checkpoints/gpt2_init.pt
if [ ! -f "$INIT_PATH" ]; then
    echo "=== $(date) downloading GPT-2 small and copying weights ==="
    python -u model/init_from_gpt2.py --output "$INIT_PATH" 2>&1 | tee runs/gpt2_init.log
else
    echo "=== $(date) reusing existing $INIT_PATH ==="
fi

echo "=== $(date) starting $RUN_NAME ==="
echo "python: $(which python)"
echo "wandb run: $WANDB_NAME (group=$WANDB_RUN_GROUP, dir=$WANDB_DIR)"

cd model
python -u train.py \
    --preset 3090 \
    --model-size 125M \
    --resume ../"$INIT_PATH" \
    --synthetic-jsonl data/train.jsonl \
    --rag-context-prob 0.5 \
    --learning-rate 1e-4 \
    --warmup-steps 100 \
    --dropout 0.1 \
    --label-smoothing 0.05 \
    --ema-decay 0.999 \
    --early-stopping-patience 3 \
    --use-wandb \
    --log-curves "../runs/curves_${RUN_NAME}.json" \
    --out-dir "../checkpoints/${RUN_NAME}" \
    2>&1 | tee "../runs/run_${RUN_NAME}.log"

echo "=== $(date) done ==="
