"""Freeze the held-out eval sets (all from HumanEval, never used in training).

coding : all 164 HumanEval prompts, first pass.
repair : the base model's own failed HumanEval drafts + single-bug mutants of
         canonical solutions, each with real test feedback in context.
review : balanced buggy / passing code; the model must give FAILURE_PROBABILITY.
tools  : tools_data.EVAL (partly unseen tools).
Needs work/base_humaneval_greedy.jsonl (collect_base.py humaneval).
"""
import random
import mutate, sandbox, tasks
from schema import ROOT, WORK, jsonl, write_jsonl

rng = random.Random(1234)
he = {t["id"]: t for t in tasks.humaneval()}
base = {r["id"]: r for r in jsonl(WORK / "base_humaneval_greedy.jsonl")}
ids = sorted(he)
rng.shuffle(ids)

fails = [i for i in ids if not base[i]["passed"]]
passes = [i for i in ids if base[i]["passed"]]


def one_mutant(t, src, seed):
    for m in mutate.mutants(src, seed=seed):
        ok, fb = sandbox.run(t, m["code"])
        if not ok and not fb.startswith("TimeoutError"):
            return m, fb
    return None, None


# ---- repair: base failures (on-policy drafts) + mutants on a disjoint task slice
repair = []
for i in fails:
    repair.append({"id": f"repair/base/{i}", "task_id": i, "source": "base_draft",
                   "draft_response": base[i]["response"], "draft_code": base[i]["code"],
                   "feedback": base[i]["feedback"]})
mut_ids, rev_ids = passes[:len(passes) // 2], passes[len(passes) // 2:]
n_mut = 0
for i in mut_ids:
    if n_mut >= 45:
        break
    t = he[i]
    m, fb = one_mutant(t, t["reference"], seed=n_mut)
    if m:
        repair.append({"id": f"repair/mut/{i}", "task_id": i, "source": "mutant", "op": m["op"],
                       "draft_response": f"```python\n{m['code'].strip()}\n```", "draft_code": m["code"],
                       "feedback": fb})
        n_mut += 1

# ---- review: balanced; tasks disjoint from the mutant-repair slice
review = []
bug_base = fails[: 20]
for i in bug_base:
    review.append({"id": f"review/base_bug/{i}", "task_id": i, "code": base[i]["code"], "buggy": True})
k = 0
for i in rev_ids:
    if k >= 20:
        break
    m, fb = one_mutant(he[i], he[i]["reference"], seed=100 + k)
    if m:
        review.append({"id": f"review/mut/{i}", "task_id": i, "code": m["code"], "buggy": True})
        k += 1
good_pool = rev_ids[:]
rng.shuffle(good_pool)
for i in good_pool[:20]:
    review.append({"id": f"review/base_ok/{i}", "task_id": i, "code": base[i]["code"], "buggy": False})
for i in good_pool[20:40]:
    review.append({"id": f"review/ref_ok/{i}", "task_id": i, "code": he[i]["reference"], "buggy": False})

write_jsonl(ROOT / "data/eval_repair.jsonl", repair)
write_jsonl(ROOT / "data/eval_review.jsonl", review)
print(f"repair {len(repair)} ({len(fails)} base drafts, {n_mut} mutants); "
      f"review {len(review)} ({sum(r['buggy'] for r in review)} buggy)")
