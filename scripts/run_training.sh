#!/usr/bin/env bash
# Launcher used by `screen -dmS train bash scripts/run_training.sh`.
# Sets shared-disk caches, activates the venv, and starts the 25M run on the
# fixed pipeline.  Output is tee'd to runs/run_25m_v2.log so we can tail it
# without attaching to the screen session.
set -u

cd "$(dirname "$0")/.."

export TMPDIR=/mnt/obelix_serving/tmp
export PIP_CACHE_DIR=/mnt/obelix_serving/pip-cache
export HF_HOME=/mnt/obelix_serving/hf_cache
export TRANSFORMERS_CACHE=/mnt/obelix_serving/hf_cache
export HF_DATASETS_CACHE=/mnt/obelix_serving/hf_cache
# Pin tokenizer + torch.compile caches to obelix_serving too so the small
# system partition never fills.
export TIKTOKEN_CACHE_DIR=/mnt/obelix_serving/hf_cache/tiktoken
export TORCHINDUCTOR_CACHE_DIR=/mnt/obelix_serving/tmp/torchinductor
mkdir -p "$TMPDIR" "$TIKTOKEN_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

source .venv/bin/activate

mkdir -p runs checkpoints/run_25m_v2

echo "=== $(date) starting run_25m_v2 ==="
echo "python: $(which python)"
nvidia-smi | head -10

cd model
python train.py \
    --preset 3090 \
    --model-size 25M \
    --synthetic-jsonl data/train.jsonl \
    --rag-context-prob 0.5 \
    --log-curves ../runs/curves_25m_v2.json \
    --out-dir ../checkpoints/run_25m_v2 \
    --early-stopping-patience 5 \
    2>&1 | tee ../runs/run_25m_v2.log

echo "=== $(date) done ==="
