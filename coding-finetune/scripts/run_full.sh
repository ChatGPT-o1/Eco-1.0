#!/bin/sh
# Full run: train (2 epochs, epoch-1 checkpoint kept), then full held-out evals and the gate.
cd "$(dirname "$0")"
PY=../work/venv/bin/python
export MALLOC_ARENA_MAX=2
$PY train.py --data ../data/train.jsonl --out ../work/lora_full --lr 5e-5 --epochs 2 > ../work/log_full_train.txt 2>&1 || exit 1
echo TRAIN_DONE
$PY evaluate.py --name tuned_ep2 --adapter ../work/lora_full --batch 16 > ../work/log_eval_tuned_ep2.txt 2>&1 || exit 1
echo EVAL_TUNED_DONE
$PY evaluate.py --name base --batch 16 > ../work/log_eval_base.txt 2>&1 || exit 1
echo EVAL_BASE_DONE
$PY promote.py base tuned_ep2 > ../work/log_gate_ep2.txt 2>&1
echo GATE_DONE
