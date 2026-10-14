#!/usr/bin/env bash
cd "$(dirname "$0")/.."
./venv/bin/python gen.py --adapter lora_chat_v3 --input fix/eval.jsonl --out fix/eval_v3.jsonl --max_new 100 --batch 20 > fix/eval_v3.log 2>&1
./venv/bin/python gen.py --adapter lora_chat_v2 --input fix/eval.jsonl --out fix/eval_v2.jsonl --max_new 100 --batch 20 > fix/eval_v2.log 2>&1
echo ALLDONE > fix/eval_done.flag
