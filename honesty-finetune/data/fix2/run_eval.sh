#!/usr/bin/env bash
cd "$(dirname "$0")/.."
./venv/bin/python gen.py --adapter lora_chat_v4 --input fix2/eval2.jsonl --out fix2/eval_v4.jsonl --max_new 110 --batch 24 > fix2/eval_v4.log 2>&1
./venv/bin/python gen.py --adapter lora_chat_v3 --input fix2/eval2.jsonl --out fix2/eval_v3.jsonl --max_new 110 --batch 24 > fix2/eval_v3.log 2>&1
echo ALLDONE > fix2/eval_done.flag
