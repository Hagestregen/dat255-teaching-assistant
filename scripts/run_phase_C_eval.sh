#!/usr/bin/env bash
# Phase C of the experiment plan: evaluation pass on every Phase-A and
# Phase-B checkpoint, plus the LLM-as-judge run on the top performers.
#
# Inputs:
#   - All checkpoints/*/best_ema.pt produced by Phases A and B.
#   - eval/test_split.json — held-out split frozen earlier so every run
#     evaluates on the SAME 99 test examples.
#
# Outputs:
#   - eval/results/<run>_ema/                 metrics, no context
#   - eval/results/<run>_ema_ctx/             metrics, with retrieved context
#   - eval/results/<run>_ema_judge/           LLM-judge scores (top runs only)
#
# Usage:
#     bash scripts/run_phase_C_eval.sh
set -u

cd "$(dirname "$0")/.."

export TMPDIR=/mnt/obelix_serving/tmp
export HF_HOME=/mnt/obelix_serving/hf_cache
export TRANSFORMERS_CACHE=/mnt/obelix_serving/hf_cache
export HF_DATASETS_CACHE=/mnt/obelix_serving/hf_cache
export TIKTOKEN_CACHE_DIR=/mnt/obelix_serving/hf_cache/tiktoken

source .venv/bin/activate

# Pick checkpoint AND decide whether --use-ema is safe.
# Echoes "<ckpt_path>|<use_ema 0/1>".
#
# For from-scratch runs the EMA is well-formed and best_ema.pt is preferred.
# For GPT-2-warm-start runs the EMA gets polluted by the initial loss spike
# (where the freshly-random task-token embeddings cause loss ~30 at step 20).
# That spike takes thousands of steps to decay out of the EMA so the EMA
# weights are worse than the live weights at every val we'll see.  For those
# runs we use best.pt without --use-ema.
pick_ckpt_and_mode() {
    local d="$1"
    local ckpt=""
    local use_ema=0
    case "$d" in
        gpt2_warm_*)
            # Warm-start: live best.pt + no EMA
            if [ -f "checkpoints/$d/best.pt" ]; then
                ckpt="checkpoints/$d/best.pt"
                use_ema=0
            fi
            ;;
        *)
            # From-scratch: EMA weights are healthy
            if [ -f "checkpoints/$d/best_ema.pt" ]; then
                ckpt="checkpoints/$d/best_ema.pt"
                use_ema=1
            elif [ -f "checkpoints/$d/best.pt" ]; then
                ckpt="checkpoints/$d/best.pt"
                use_ema=0
            fi
            ;;
    esac
    echo "${ckpt}|${use_ema}"
}

eval_one() {
    local run="$1"; shift
    local suffix="$1"; shift
    local extra=( "$@" )
    local pick
    pick=$(pick_ckpt_and_mode "$run")
    local ckpt="${pick%|*}"
    local use_ema="${pick#*|}"
    if [ -z "$ckpt" ]; then
        echo "SKIP $run: no best.pt or best_ema.pt"
        return
    fi

    local outdir="eval/results/${run}_${suffix}"
    if [ -f "$outdir/summary.json" ]; then
        echo "=== skipping $run/$suffix (already evaluated) ==="
        return
    fi
    mkdir -p "$outdir"
    echo "=== $(date) eval $run → $outdir (use_ema=$use_ema) ==="
    local ema_arg=()
    [ "$use_ema" = "1" ] && ema_arg=(--use-ema)
    python -u eval/run_eval.py \
        --checkpoint "$ckpt" \
        --output-dir "$outdir" \
        "${ema_arg[@]}" \
        --max-examples 0 \
        "${extra[@]}" \
        2>&1 | tee "$outdir/run.log"
}

# Phase A + B runs we want to score.  Trimmed to 3 runs under the 3-hour
# total time budget on 2026-04-29.  Eval order = most informative first so
# the script can be killed early without losing the headline pair.
RUNS=(
    "gpt2_warm_full"   # A2, headline warm-start run (most important)
    "scratch_125m"     # A1, from-scratch 125M baseline (warm-start contrast)
    "gpt2_warm_rag10"  # B1, always-context training (RAG ablation)
)

# Pass 1: metric-only eval, no context (all 3 runs).
for run in "${RUNS[@]}"; do
    eval_one "$run" "ema"
done

# Pass 2: with retrieved context at inference — but ONLY for the runs that
# benefit (headline + RAG ablation).  The from-scratch baseline doesn't
# have the RAG distribution baked in so its with-context numbers aren't
# meaningful, and skipping it saves ~10 min.
for run in "gpt2_warm_full" "gpt2_warm_rag10"; do
    eval_one "$run" "ema_ctx" --with-context
done

# LLM-judge step intentionally OMITTED under the 3-hour budget.  BERTScore
# (semantic) + METEOR (synonym-aware) + ROUGE-L are sufficient for the
# report's results table; W&B sample tables provide the qualitative side.

echo "=== $(date) Phase C eval complete ==="
