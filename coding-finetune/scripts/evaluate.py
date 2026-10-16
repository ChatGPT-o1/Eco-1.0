"""Held-out evaluation: first-pass coding, repair, self-review calibration, tools, voice.

usage: evaluate.py --name base [--adapter PATH] [--frac 1.0]
Writes results/<name>.json (scores) and work/eval_<name>.jsonl (generations).
"""
import argparse
import json
import random
import time
import gen, sandbox, tasks
import tools_data as td
from schema import (ROOT, WORK, code_prompt, error_turn, extract_code, filler_hits, jsonl, parse_probability,
                    parse_repair, parse_tool_calls, prose_outside_code, review_prompt, is_duplicate, task_text,
                    write_jsonl)

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--adapter")
ap.add_argument("--frac", type=float, default=1.0, help="deterministic subsample for smoke tests")
ap.add_argument("--only", default="coding,repair,review,tools")
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()
only = set(args.only.split(","))


def sub(rows):
    if args.frac >= 1:
        return rows
    r = random.Random(7)
    return sorted(r.sample(rows, max(4, int(len(rows) * args.frac))), key=lambda x: str(x.get("id")))


def log(m):
    print(m, flush=True)


he = {t["id"]: t for t in tasks.humaneval()}
model, tok = gen.load(args.adapter)
scores, dump = {"name": args.name, "adapter": args.adapter, "frac": args.frac}, []
t0 = time.time()


def voice(texts):
    n = len(texts)
    pre, post = zip(*[prose_outside_code(t) for t in texts]) if texts else ((), ())
    return {"filler_rate": sum(bool(filler_hits(t)) for t in texts) / max(n, 1),
            "mean_words_before_code": sum(pre) / max(n, 1),
            "mean_words_after_code": sum(post) / max(n, 1)}


# ---------------- coding
if "coding" in only:
    items = sub(sorted(he.values(), key=lambda t: t["id"]))
    outs = gen.generate(model, tok, [{"messages": [{"role": "user", "content": code_prompt(t)}]} for t in items],
                        max_new=1024, batch=args.batch, log=log)
    codes = [extract_code(o[0], t["entry_point"]) for o, t in zip(outs, items)]
    res = sandbox.run_many(list(zip(items, codes)))
    ok = [r[0] for r in res]
    scores["coding"] = {"pass": sum(ok), "n": len(ok), "rate": sum(ok) / len(ok),
                        "truncated": sum(not o[1] for o in outs), **voice([o[0] for o in outs])}
    dump += [{"part": "coding", "id": t["id"], "out": o[0], "passed": r[0], "feedback": r[1]}
             for t, o, r in zip(items, outs, res)]
    log(f"coding {scores['coding']} {time.time() - t0:.0f}s")

# ---------------- repair
if "repair" in only:
    items = sub(jsonl(ROOT / "data/eval_repair.jsonl"))
    msgs = [{"messages": [{"role": "user", "content": code_prompt(he[r["task_id"]])},
                          {"role": "assistant", "content": r["draft_response"]},
                          {"role": "user", "content": error_turn(r["feedback"])}]} for r in items]
    outs = gen.generate(model, tok, msgs, max_new=1024, batch=args.batch, log=log)
    codes = [extract_code(o[0], he[r["task_id"]]["entry_point"]) for o, r in zip(outs, items)]
    res = sandbox.run_many([(he[r["task_id"]], c) for r, c in zip(items, codes)])
    dup = [is_duplicate(c, r["draft_code"], he[r["task_id"]]["entry_point"]) for c, r in zip(codes, items)]
    fp = [all(parse_repair(o[0])) for o in outs]
    scores["repair"] = {"pass": sum(x[0] for x in res), "n": len(items), "rate": sum(x[0] for x in res) / len(items),
                        "duplicate_of_failed": sum(dup), "fault_pivot_format": sum(fp) / len(items),
                        **{k: sum(x[0] for x, r in zip(res, items) if r["source"] == k) for k in ("base_draft", "mutant")},
                        **voice([o[0] for o in outs])}
    dump += [{"part": "repair", "id": r["id"], "out": o[0], "passed": x[0], "feedback": x[1], "duplicate": d}
             for r, o, x, d in zip(items, outs, res, dup)]
    log(f"repair {scores['repair']} {time.time() - t0:.0f}s")

# ---------------- self-review calibration
if "review" in only:
    items = sub(jsonl(ROOT / "data/eval_review.jsonl"))
    msgs = [{"messages": [{"role": "user", "content": review_prompt(task_text(he[r["task_id"]]), r["code"])}]}
            for r in items]
    outs = gen.generate(model, tok, msgs, max_new=320, batch=args.batch, log=log)
    probs = [parse_probability(o[0]) for o in outs]
    y = [r["buggy"] for r in items]
    parsed = [p is not None for p in probs]
    pf = [0.5 if p is None else p for p in probs]  # unparsed counts as uninformative
    pos = [p for p, b in zip(pf, y) if b]
    neg = [p for p, b in zip(pf, y) if not b]
    auroc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(len(pos) * len(neg), 1)
    acc = sum(((p is not None) and ((p >= 0.5) == b)) for p, b in zip(probs, y)) / len(y)
    scores["review"] = {"n": len(y), "parse_rate": sum(parsed) / len(y), "auroc": auroc, "acc_at_50": acc,
                        "detected_buggy": sum((p is not None and p >= 0.5) for p, b in zip(probs, y) if b),
                        "n_buggy": sum(y),
                        "mean_p_buggy": sum(pos) / max(len(pos), 1), "mean_p_ok": sum(neg) / max(len(neg), 1),
                        "zero_on_buggy": sum((p is not None and p == 0) for p, b in zip(probs, y) if b),
                        "brier": sum((p - b) ** 2 for p, b in zip(pf, y)) / len(y), **voice([o[0] for o in outs])}
    dump += [{"part": "review", "id": r["id"], "out": o[0], "p": p, "buggy": r["buggy"]}
             for r, o, p in zip(items, outs, probs)]
    log(f"review {scores['review']} {time.time() - t0:.0f}s")


# ---------------- tools
def match(v, want):
    if isinstance(want, dict) and "$any" in want:
        return v not in (None, "", [], {})
    if isinstance(want, dict) and "$contains" in want:
        return isinstance(v, str) and want["$contains"] in v
    if isinstance(want, dict) and "$in" in want:
        return v in want["$in"]
    if isinstance(want, str) and isinstance(v, str):
        return " ".join(v.split()) == " ".join(want.split())
    return v == want


if "tools" in only:
    items = td.EVAL if args.frac >= 1 else td.EVAL[::2]
    outs = gen.generate(model, tok, [{"messages": c["messages"], "tools": c["tools"]} for c in items],
                        max_new=320, batch=args.batch, log=log)
    ok, fmt = [], []
    for c, o in zip(items, outs):
        calls = parse_tool_calls(o[0], c["tools"])
        exp = c["expect"]
        stray = "<tool_call>" in o[0] and not calls
        fmt.append(not stray)
        if exp["name"] is None:
            ok.append(not calls and "<tool_call>" not in o[0] and len(o[0].strip()) > 0)
            continue
        if len(calls) != 1:
            ok.append(False)
            continue
        cl = calls[0]
        good = match(cl["name"], exp["name"])
        want = dict(exp["arguments"])
        want.update(exp.get("if_name", {}).get(cl["name"], {}))
        good = good and all(match(cl["arguments"].get(k), v) for k, v in want.items())
        valid = {p for p in next(t["function"]["parameters"]["properties"] for t in c["tools"]
                                 if t["function"]["name"] == cl["name"])} if any(
            t["function"]["name"] == cl["name"] for t in c["tools"]) else set()
        good = good and set(cl["arguments"]) <= valid
        ok.append(good)
    scores["tools"] = {"pass": sum(ok), "n": len(ok), "rate": sum(ok) / len(ok), "format_ok": sum(fmt) / len(fmt)}
    dump += [{"part": "tools", "id": c["id"], "out": o[0], "passed": k} for c, o, k in zip(items, outs, ok)]
    log(f"tools {scores['tools']} {time.time() - t0:.0f}s")

scores["seconds"] = round(time.time() - t0)
write_jsonl(WORK / f"eval_{args.name}.jsonl", dump)
(ROOT / "results").mkdir(exist_ok=True)
json.dump(scores, open(ROOT / f"results/{args.name}.json", "w"), indent=1)
log(json.dumps(scores, indent=1))
