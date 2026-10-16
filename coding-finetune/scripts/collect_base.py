"""Run the base model greedy on (a) the MBPP train pool or (b) HumanEval.
(a) gives on-policy drafts: passing ones feed first-pass targets, failing ones
    feed repair trajectories. (b) gives the held-out repair/review items.
Writes incrementally (resumable).  usage: collect_base.py mbpp|humaneval [batch]
"""
import json
import sys
import gen, sandbox, tasks
from schema import WORK, code_prompt, extract_code, jsonl

which = sys.argv[1]
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 16
pool = tasks.mbpp() if which == "mbpp" else tasks.humaneval()
path = WORK / f"base_{which}_greedy.jsonl"
done = {r["id"] for r in jsonl(path)} if path.exists() else set()
todo = sorted((t for t in pool if t["id"] not in done), key=lambda t: len(code_prompt(t)))
model, tok = gen.load()
for s in range(0, len(todo), batch * 2):
    chunk = todo[s:s + batch * 2]
    outs = gen.generate(model, tok, [{"messages": [{"role": "user", "content": code_prompt(t)}]} for t in chunk],
                        max_new=640, batch=batch)
    codes = [extract_code(o[0], t["entry_point"]) for o, t in zip(outs, chunk)]
    res = sandbox.run_many(list(zip(chunk, codes)))
    with open(path, "a") as f:
        for t, o, c, (ok, fb) in zip(chunk, outs, codes, res):
            f.write(json.dumps({"id": t["id"], "response": o[0], "finished": o[1], "code": c, "passed": ok,
                                "feedback": fb}, ensure_ascii=False) + "\n")
    rows = jsonl(path)
    print(f"{len(rows)}/{len(pool)} pass {sum(r['passed'] for r in rows)}", flush=True)
rows = jsonl(path)
print(which, "pass", sum(r["passed"] for r in rows), "/", len(rows), flush=True)
