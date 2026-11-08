#!/bin/bash
# Train GrandeRL-1.0 on SmolLM2-135M-Instruct for MBPP, then score it and the starting model on 500 MBPP test + 164 HumanEval.
# Safe to re-run: training resumes from its last checkpoint, finished evaluations are skipped.
#
#   ./run_granderl.sh            # 120 compute-minutes of training (default)
#   ./run_granderl.sh 60         # shorter run
#   THREADS=2 EXTRA="--no-teacher 1" ./run_granderl.sh
set -u
D="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3} -I"
BUDGET=${1:-120}
export ENGINE_MODEL=${ENGINE_MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}
cd "$D/crl/v8"
mkdir -p "$D/runs" "$D/evals"
until [ -f "$D/runs/granderl/DONE" ]; do
  $PY train.py --method granderl --out "$D/runs/granderl" --budget-min "$BUDGET" --threads "${THREADS:-4}" --micro 4 ${EXTRA:-} \
      >> "$D/runs/granderl.log" 2>&1 || sleep 10
done
[ -f "$D/evals/base.json" ] || $PY evalx.py --model "$ENGINE_MODEL" --out "$D/evals/base.json" --threads "${THREADS:-4}" >> "$D/evals/base.log" 2>&1
[ -f "$D/evals/granderl.json" ] || $PY evalx.py --model "$D/runs/granderl/final" --out "$D/evals/granderl.json" --threads "${THREADS:-4}" >> "$D/evals/granderl.log" 2>&1
$PY compare.py "$D/evals/base.json" "$D/evals/granderl.json" | tee "$D/evals/compare.txt"
