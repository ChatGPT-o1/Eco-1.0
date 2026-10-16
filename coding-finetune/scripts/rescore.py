"""Re-extract and re-run base drafts after an extractor change. usage: rescore.py mbpp|humaneval"""
import sys
import sandbox, tasks
from schema import WORK, extract_code, jsonl, write_jsonl
which = sys.argv[1]
pool = {t["id"]: t for t in (tasks.mbpp() if which == "mbpp" else tasks.humaneval())}
path = WORK / f"base_{which}_greedy.jsonl"
rows = jsonl(path)
before = sum(r["passed"] for r in rows)
for r in rows:
    r["code"] = extract_code(r["response"], pool[r["id"]]["entry_point"])
res = sandbox.run_many([(pool[r["id"]], r["code"]) for r in rows])
for r, (ok, fb) in zip(rows, res):
    r["passed"], r["feedback"] = ok, fb
write_jsonl(path, rows)
print(which, f"pass {before} -> {sum(r['passed'] for r in rows)} / {len(rows)}")
