#!/usr/bin/env bash
cd "$(dirname "$0")"
until grep -q -E "SAVED|Traceback|Killed" train4.log; do sleep 30; done
./venv/bin/python gen.py --adapter lora_honesty_v4 --input eval_prompts.jsonl insist_eval_ext.jsonl --out v4_gen.jsonl > v4.log 2>&1
echo ALLDONE > v4_done.flag
