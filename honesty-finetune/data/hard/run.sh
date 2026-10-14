#!/usr/bin/env bash
# resumable: for each model, generate only prompts not yet in its output file, one batch at a time
cd "$(dirname "$0")/.."
for spec in "base:" "v7:lora_honesty_v7" "chat_v1:lora_chat_v1"; do
  name="${spec%%:*}"; ad="${spec#*:}"; out="hard/out_${name}.jsonl"; touch "$out"
  while true; do
    ./venv/bin/python -I - "$out" <<'PY' > hard/todo.jsonl
import json, sys
done = {json.loads(l)["id"] for l in open(sys.argv[1]) if l.strip()}
todo = [l for l in open("hard/prompts.jsonl") if json.loads(l)["id"] not in done][:8]
sys.stdout.write("".join(todo))
PY
    [ -s hard/todo.jsonl ] || break
    if [ -n "$ad" ]; then A="--adapter $ad"; else A=""; fi
    ./venv/bin/python gen.py $A --input hard/todo.jsonl --out hard/part.jsonl --max_new 200 --batch 8 > hard/gen_${name}.log 2>&1 || exit 1
    cat hard/part.jsonl >> "$out"
  done
done
echo ALLDONE > hard/done.flag
