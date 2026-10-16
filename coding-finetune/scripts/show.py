"""Print MBPP base drafts compactly for hand annotation (re-extracts with the current extractor).
usage: show.py fail|pass START N [--skip-done]"""
import sys
import clean, sandbox, tasks
from schema import ROOT, WORK, extract_code, jsonl
mode, start, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pool = {t["id"]: t for t in tasks.mbpp()}
rows = jsonl(WORK / "base_mbpp_greedy.jsonl")
for r in rows:
    r["code"] = extract_code(r["response"], pool[r["id"]]["entry_point"])
res = sandbox.run_many([(pool[r["id"]], r["code"]) for r in rows])
for r, (ok, fb) in zip(rows, res):
    r["passed"], r["feedback"] = ok, fb
import annotations as A
done = {a["id"] for a in A.REPAIRS} | set(A.SKIP) | set(A.FRAMING)
rows = [r for r in rows if r["passed"] == (mode == "pass") and r["id"] not in done]
print(f"[{len(rows)} remaining in {mode}]")
for r in rows[start:start + n]:
    t = pool[r["id"]]
    print(f"##### {r['id']}  {t['text']}")
    if mode == "fail":
        print("TESTS:", *t["tests"], sep="\n  ")
        print("CODE:\n" + clean.clean_code(r["code"])[:1800])
        print("FEEDBACK:", r["feedback"][-400:])
    else:
        print(clean.clean_code(r["code"])[:1200])
