"""Verify every annotation applies to its draft and passes the tests (current extractor)."""
import clean, sandbox, tasks
from annotations import REPAIRS
from schema import WORK, extract_code, jsonl
pool = {t["id"]: t for t in tasks.mbpp()}
base = {r["id"]: r for r in jsonl(WORK / "base_mbpp_greedy.jsonl")}
bad = 0
for a in REPAIRS:
    r = base.get(a["id"])
    if r is None:
        print("NO DRAFT", a["id"]); bad += 1; continue
    draft = extract_code(r["response"], pool[a["id"]]["entry_point"])
    if "code" in a:
        new = a["code"]
    else:
        new = draft
        for old, rep in a["edits"]:
            if new.count(old) != 1:
                print("EDIT MISS", a["id"], repr(old)); new = None; break
            new = new.replace(old, rep, 1)
    if new is None:
        bad += 1; continue
    ok, fb = sandbox.run(pool[a["id"]], clean.clean_code(new))
    d_ok, _ = sandbox.run(pool[a["id"]], draft)
    if not ok or d_ok:
        bad += 1
        print("FAIL" if not ok else "DRAFT ALREADY PASSES", a["id"], fb[:300])
print(f"{len(REPAIRS)} annotations, {bad} problems")
