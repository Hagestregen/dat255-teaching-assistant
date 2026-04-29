#!/usr/bin/env bash
# Phase A of the experiment plan (revised): ablations that justify the
# headline result, all trained on the FULL train split.
#
#   A1 — 125M model trained FROM SCRATCH (no GPT-2 init).
#        Pair this with the warm-start baseline to quantify the warm-start gain.
#
#   A2 — gpt2_warm baseline on the FULL train split.  This is the headline
#        run: GPT-2 small initialisation, mild regularisation, all data.
#        Every other ablation is compared against this run.
#
# (The earlier 10/25/50 % data-size sweep was removed by request — every
# experiment now sees all available training data so comparisons are over
# identical training distributions.)
#
# Both runs are sequenced in a single screen session so the GPU is never
# idle between them.  Every run is keyed off RUN_NAME so curves JSON,
# checkpoint dir, W&B run name and the experiment registry agree.
#
# Usage (inside screen):
#     bash scripts/run_phase_A.sh
set -u

cd "$(dirname "$0")/.."

# -- shared environment ------------------------------------------------------
export TMPDIR=/mnt/obelix_serving/tmp
export PIP_CACHE_DIR=/mnt/obelix_serving/pip-cache
export HF_HOME=/mnt/obelix_serving/hf_cache
export TRANSFORMERS_CACHE=/mnt/obelix_serving/hf_cache
export HF_DATASETS_CACHE=/mnt/obelix_serving/hf_cache
export TIKTOKEN_CACHE_DIR=/mnt/obelix_serving/hf_cache/tiktoken
export TORCHINDUCTOR_CACHE_DIR=/mnt/obelix_serving/tmp/torchinductor
export WANDB_DIR=/mnt/obelix_serving/wandb_runs
mkdir -p "$TMPDIR" "$TIKTOKEN_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_DIR"

source .venv/bin/activate
mkdir -p runs

INIT_PATH=checkpoints/gpt2_init.pt
if [ ! -f "$INIT_PATH" ]; then
    echo "=== $(date) downloading GPT-2 small and copying weights ==="
    python -u model/init_from_gpt2.py --output "$INIT_PATH" \
        2>&1 | tee runs/gpt2_init.log
else
    echo "=== $(date) reusing existing $INIT_PATH ==="
fi

# -- helper: run one training job -------------------------------------------
# Args: 1 = run_name, 2 = wandb_group, then the rest are extra train.py flags.
run_one() {
    local name="$1" ; shift
    local group="$1"; shift
    local extra=( "$@" )

    # Resume-safe: re-invoking the script skips runs whose log already shows
    # "Training complete!".  Lets us pause the screen mid-phase and pick up
    # cleanly the next day without retraining finished runs.
    local logfile="runs/run_${name}.log"
    if [ -f "$logfile" ] && grep -q "Training complete!" "$logfile"; then
        echo "=== skipping $name (already finished — log shows 'Training complete!') ==="
        return
    fi

    export WANDB_NAME="$name"
    export WANDB_RUN_GROUP="$group"

    mkdir -p "checkpoints/$name"
    echo "=========================================================="
    echo "=== $(date) starting $name (group=$group) ==="
    echo "=== extra flags: ${extra[*]} ==="
    echo "=========================================================="

    (
        cd model
        python -u train.py \
            --preset 3090 \
            --model-size 125M \
            --splits-dir data/splits \
            --rag-context-prob 0.5 \
            --dropout 0.1 \
            --label-smoothing 0.05 \
            --ema-decay 0.999 \
            --early-stopping-patience 5 \
            --use-wandb \
            --log-curves "../runs/curves_${name}.json" \
            --out-dir "../checkpoints/${name}" \
            "${extra[@]}" \
            2>&1 | tee "../runs/run_${name}.log"
    )

    echo "=== $(date) finished $name ==="
    # Give the CUDA driver a few seconds to release CUBLAS handles before
    # the next run starts.  Without this we sometimes see
    # CUBLAS_STATUS_NOT_INITIALIZED on the first forward of the next run.
    sleep 10
}

# ---------------------------------------------------------------------------
# A1: 125M from scratch (no resume).  Use the same lr/warmup as the
# from-scratch 25M run so it isn't unfairly handicapped — the larger model
# tolerates the standard schedule.  This is the "no-warm-start" baseline.
# ---------------------------------------------------------------------------
run_one "scratch_125m" "ablation_init"

# ---------------------------------------------------------------------------
# A2: GPT-2 warm-start headline run on the full train split.  Every other
# experiment is compared against this row in the report's results table.
# ---------------------------------------------------------------------------
run_one "gpt2_warm_full" "headline" \
    --resume "../$INIT_PATH" \
    --learning-rate 1e-4 --warmup-steps 100

echo "=== $(date) Phase A complete ==="
