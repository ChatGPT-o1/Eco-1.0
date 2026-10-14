#!/usr/bin/env bash
cd "$(dirname "$0")"
./venv/bin/python gen.py --input insist_eval_ext.jsonl --out base_ext_gen.jsonl > base_ext.log 2>&1
./venv/bin/python gen.py --adapter lora_honesty_v2 --input insist_eval_ext.jsonl --out v2_ext_gen.jsonl > v2_ext.log 2>&1
./venv/bin/python gen.py --adapter lora_honesty_v3 --input eval_prompts.jsonl insist_eval_ext.jsonl --out v3_gen.jsonl > v3.log 2>&1
echo ALLDONE > evals_done.flag
