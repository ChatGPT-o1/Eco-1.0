#!/usr/bin/env bash
set -u
DEST="$1"
BASE="https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main"
FILES=$(python3 -I -c '
import json
for f in json.load(open("/tmp/hf_tree.json")):
    if f.get("type") == "file":
        print(f["path"])
')
fail=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  out="$DEST/$f"
  mkdir -p "$(dirname "$out")"
  if curl -fL --retry 4 --retry-delay 5 -C - --max-time 3600 -sS -o "$out" "$BASE/$f"; then
    echo "OK   $f"
  else
    echo "FAIL $f"; fail=1
  fi
done <<< "$FILES"
echo "DONE fail=$fail"
