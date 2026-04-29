#!/usr/bin/env bash
# Phase B of the experiment plan: ablations that probe the recipe details.
#
#   B1 — RAG context-mixing ablation.  Same gpt2_warm recipe as the headline
#        run, but with --rag-context-prob = 0.0 (never show context) and 1.0
#        (always show context).  Pair with gpt2_warm_full (= 0.5) to see how
#        robust the model is to the with/without-context distribution.
#
#   B2 — Architecture variant: RMSNorm + SwiGLU.  Cannot warm-start from
#        GPT-2 (different blocks), so we train from scratch and pair with
#        the A1 result (scratch_125m, LayerNorm + GELU) for a clean compare.
#
# Run AFTER Phase A finishes so we don't interleave compute / W&B groups.
#
# Usage (inside screen):
#     bash scripts/run_phase_B.sh
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
    echo "ERROR: $INIT_PATH missing.  Run Phase A first or rerun init_from_gpt2.py." >&2
    exit 1
fi

run_one() {
    local name="$1" ; shift
    local group="$1"; shift
    local extra=( "$@" )

    # Resume-safe: if a previous run for this name reached "Training complete!"
    # we skip it.  Lets us re-invoke this script after an interruption and
    # only the unfinished tail re-runs.  Detection is based on the log marker
    # because curves_*.json is also written for crashed runs.
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
    sleep 10  # let CUDA release CUBLAS handles before the next run
}

# ---------------------------------------------------------------------------
# B1 (reduced): RAG context-mix ablation.  We only run the "always-context"
# variant (rag-prob = 1.0) because it contrasts cleanly with the headline
# run's mixed (0.5) training distribution.  The symmetric "never-context"
# (rag-prob = 0.0) run was dropped under time pressure on 2026-04-29 — it
# was nearly identical to 0.5 in earlier v1-data runs, so the report does
# not lose a meaningful story by skipping it.
# (RMSNorm+SwiGLU architecture variant also dropped for the same reason.)
# ---------------------------------------------------------------------------
run_one "gpt2_warm_rag10" "rag_ablation" \
    --resume "../$INIT_PATH" \
    --learning-rate 1e-4 --warmup-steps 100 \
    --rag-context-prob 1.0

echo "=== $(date) Phase B complete ==="
