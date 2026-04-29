#!/usr/bin/env bash
# v4: GPT-2 small (124M) warm start + fine-tune on the QA pipeline.
# Two phases inside this single script:
#   1. download GPT-2 weights into checkpoints/gpt2_init.pt (only the first run)
#   2. resume training from that checkpoint with mild regularization
#
# Caches are pinned to /mnt/obelix_serving so the system disk stays safe.
set -u

cd "$(dirname "$0")/.."

export TMPDIR=/mnt/obelix_serving/tmp
export PIP_CACHE_DIR=/mnt/obelix_serving/pip-cache
export HF_HOME=/mnt/obelix_serving/hf_cache
export TRANSFORMERS_CACHE=/mnt/obelix_serving/hf_cache
export HF_DATASETS_CACHE=/mnt/obelix_serving/hf_cache
export TIKTOKEN_CACHE_DIR=/mnt/obelix_serving/hf_cache/tiktoken
export TORCHINDUCTOR_CACHE_DIR=/mnt/obelix_serving/tmp/torchinductor
mkdir -p "$TMPDIR" "$TIKTOKEN_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

source .venv/bin/activate
mkdir -p runs checkpoints/run_gpt2_warm

INIT_PATH=checkpoints/gpt2_init.pt

if [ ! -f "$INIT_PATH" ]; then
    echo "=== $(date) downloading GPT-2 small and copying weights ==="
    python -u model/init_from_gpt2.py --output "$INIT_PATH" 2>&1 | tee runs/gpt2_init.log
else
    echo "=== $(date) reusing existing $INIT_PATH ==="
fi

echo "=== $(date) starting run_gpt2_warm ==="
echo "python: $(which python)"

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
    --log-curves ../runs/curves_gpt2_warm.json \
    --out-dir ../checkpoints/run_gpt2_warm \
    2>&1 | tee ../runs/run_gpt2_warm.log

echo "=== $(date) done ==="
