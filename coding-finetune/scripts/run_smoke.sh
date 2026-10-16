#!/bin/sh
# Pillar 4 smoke test: 60 s of training, then a mini held-out eval of base vs smoke adapter.
cd "$(dirname "$0")"
PY=../work/venv/bin/python
$PY train.py --data ../data/train.jsonl --out ../work/lora_smoke --max_seconds 60 --lr 5e-5 > ../work/log_smoke_train.txt 2>&1 || exit 1
$PY evaluate.py --name smoke_base --frac 0.15 > ../work/log_smoke_eval_base.txt 2>&1 || exit 1
$PY evaluate.py --name smoke_lora --adapter ../work/lora_smoke --frac 0.15 > ../work/log_smoke_eval_lora.txt 2>&1 || exit 1
echo SMOKE_DONE
