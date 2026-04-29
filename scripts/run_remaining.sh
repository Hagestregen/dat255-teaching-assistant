#!/usr/bin/env bash
# scripts/run_remaining.sh
# ---------------------------------------------------------------
# Orchestrates the remaining work for the training project after
# Phase A is complete:
#   1. Phase B (single run: gpt2_warm_rag10)
#   2. Phase C evaluation (3 runs, no LLM-judge)
#   3. Registry regeneration (docs/experiments.md)
#
# Each step only proceeds if the previous one exited 0.  The eval
# step is run in any case (we still want partial metrics) but the
# registry step requires the eval summary files to exist.
#
# Usage:
#     screen -dmS remaining -L -Logfile runs/screen_remaining.log \
#         bash scripts/run_remaining.sh
# ---------------------------------------------------------------
set -u

cd "$(dirname "$0")/.."

LOG=runs/run_remaining.log
mkdir -p runs
echo "=== $(date) starting Phase B ===" | tee -a "$LOG"
bash scripts/run_phase_B.sh 2>&1 | tee -a "$LOG"
B_STATUS=${PIPESTATUS[0]}
echo "=== $(date) Phase B exited with status $B_STATUS ===" | tee -a "$LOG"

# Always run eval — even if Phase B failed we still want to score
# the runs that DID complete.
sleep 10
echo "=== $(date) starting Phase C eval ===" | tee -a "$LOG"
bash scripts/run_phase_C_eval.sh 2>&1 | tee -a "$LOG"
EVAL_STATUS=${PIPESTATUS[0]}
echo "=== $(date) Phase C eval exited with status $EVAL_STATUS ===" | tee -a "$LOG"

# Registry regen (requires eval summaries)
if [ "$EVAL_STATUS" -eq 0 ]; then
    echo "=== $(date) regenerating experiments.md ===" | tee -a "$LOG"
    source .venv/bin/activate
    python eval/compare_runs.py --out docs/experiments.md 2>&1 | tee -a "$LOG" || true
fi

echo "=== $(date) ALL DONE ===" | tee -a "$LOG"
