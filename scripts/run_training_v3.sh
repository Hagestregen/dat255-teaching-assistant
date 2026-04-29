#!/usr/bin/env bash
# v3: same 25M model, but with the regularization knobs the plan recommends
# when overfitting appears (label smoothing, stochastic depth, EMA, larger
# dropout).  All caches stay on /mnt/obelix_serving so the system disk is safe.
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

mkdir -p runs checkpoints/run_25m_v3

echo "=== $(date) starting run_25m_v3 (regularized) ==="
echo "python: $(which python)"

cd model
python -u train.py \
    --preset 3090 \
    --model-size 25M \
    --synthetic-jsonl data/train.jsonl \
    --rag-context-prob 0.5 \
    --dropout 0.3 \
    --label-smoothing 0.1 \
    --stochastic-depth 0.1 \
    --ema-decay 0.999 \
    --early-stopping-patience 3 \
    --log-curves ../runs/curves_25m_v3.json \
    --out-dir ../checkpoints/run_25m_v3 \
    2>&1 | tee ../runs/run_25m_v3.log

echo "=== $(date) done ==="
