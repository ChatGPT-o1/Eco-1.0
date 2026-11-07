#!/bin/bash
# Train V14 on SmolLM2-135M-Instruct for MBPP, then score it and the starting model on 500 MBPP test + 164 HumanEval.
# Safe to re-run: training resumes from its last checkpoint, finished evaluations are skipped.
#
#   ./run_v14.sh            # 120 compute-minutes of training (default)
#   ./run_v14.sh 60         # shorter run
#   THREADS=2 EXTRA="--no-teacher 1" ./run_v14.sh
set -u
D="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3} -I"
BUDGET=${1:-120}
export ENGINE_MODEL=${ENGINE_MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}
cd "$D/crl/v8"
mkdir -p "$D/runs" "$D/evals"
until [ -f "$D/runs/v14/DONE" ]; do
  $PY train.py --method v14 --out "$D/runs/v14" --budget-min "$BUDGET" --threads "${THREADS:-4}" --micro 4 ${EXTRA:-} \
      >> "$D/runs/v14.log" 2>&1 || sleep 10
done
[ -f "$D/evals/base.json" ] || $PY evalx.py --model "$ENGINE_MODEL" --out "$D/evals/base.json" --threads "${THREADS:-4}" >> "$D/evals/base.log" 2>&1
[ -f "$D/evals/v14.json" ] || $PY evalx.py --model "$D/runs/v14/final" --out "$D/evals/v14.json" --threads "${THREADS:-4}" >> "$D/evals/v14.log" 2>&1
$PY compare.py "$D/evals/base.json" "$D/evals/v14.json" | tee "$D/evals/compare.txt"
