#!/usr/bin/env bash
cd "$(dirname "$0")"
./venv/bin/python gen.py --adapter lora_chat_v1 --input compare_prompts.jsonl --out cmp_chat.jsonl --max_new 100 --batch 6 > cmp_chat.log 2>&1
./venv/bin/python gen.py --adapter lora_honesty_v7 --input compare_prompts.jsonl --out cmp_v7.jsonl --max_new 100 --batch 6 > cmp_v7.log 2>&1
echo ALLDONE > cmp_done.flag
