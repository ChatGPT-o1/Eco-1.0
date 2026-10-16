r"""Never-stuck inference + repair harness around Qwen3.5-2B.

    draft --check--> pass? done
                 \-> fail: one surgical repair pass (FAULT -> PIVOT -> code)
                       candidate == any failed draft (normalised AST)?  -> reject,
                       re-sample with an explicit negative constraint naming the
                       failing logic, at rising temperature (up to `resamples`).

`check` is either a test runner (code -> (passed, feedback)) or, when no tests
exist, the model's own calibrated self-review (FAILURE_PROBABILITY >= threshold
counts as a failure and its finding becomes the feedback).

CLI (eval): harness.py --name tuned --adapter work/lora_x
  re-uses the first-pass outputs from evaluate.py (work/eval_<name>.jsonl) and
  runs the repair pass on the HumanEval failures.
"""
import re
from schema import (DUPLICATE_NOTE, code_prompt, error_turn, extract_code, is_duplicate, parse_probability,
                    parse_repair, review_prompt, similarity)

NO_PROGRESS_SIM = 0.9  # a near-copy that reproduces the identical failure is also "the same code"


def culprit(feedback, code, fault=None):
    """Best short pointer at the failing logic, for the negative constraint."""
    parts = []
    lines = code.splitlines()
    for m in re.finditer(r'File "solution\.py", line (\d+)', feedback):
        n = int(m.group(1))
        if 0 < n <= len(lines):
            parts.append(f"line {n}: `{lines[n - 1].strip()}`")
    if not parts:
        first = feedback.strip().splitlines()[:2]
        parts.append(" / ".join(l.strip() for l in first))
    if fault:
        parts.append(f"(your diagnosis was: {fault})")
    return "\n".join(dict.fromkeys(parts))


class Harness:
    def __init__(self, model, tok, generate, max_repairs=1, resamples=3, review_threshold=0.5):
        self.model, self.tok, self.gen = model, tok, generate
        self.max_repairs, self.resamples, self.review_threshold = max_repairs, resamples, review_threshold

    def _one(self, messages, temperature=0.0, seed=0, max_new=640):
        return self.gen(self.model, self.tok, [{"messages": messages}], max_new=max_new,
                        batch=1, temperature=temperature, seed=seed)[0][0]

    def self_review(self, task_text, code):
        out = self._one([{"role": "user", "content": review_prompt(task_text, code)}], max_new=320)
        p = parse_probability(out)
        return (p is not None and p < self.review_threshold), out, p

    def repair(self, messages, failed, feedback, entry_point, check):
        """One surgical repair pass with the duplicate blocker.

        failed: list of (code, feedback) already known to fail.
        Returns (code, text, msgs_used, trace, passed, new_feedback).
        """
        msgs = messages + [{"role": "user", "content": error_turn(feedback)}]
        trace = []
        for attempt in range(self.resamples + 1):
            temp = 0.0 if attempt == 0 else min(0.4 + 0.2 * attempt, 1.0)
            text = self._one(msgs, temperature=temp, seed=attempt)
            code = extract_code(text, entry_point)
            dup = any(is_duplicate(code, f, entry_point) for f, _ in failed)
            ok, fb = (False, None) if dup else check(code)
            if not dup and not ok:  # no-progress: near-copy that fails exactly the same way
                dup = any(fb == ffb and similarity(code, f, entry_point) >= NO_PROGRESS_SIM for f, ffb in failed)
            trace.append({"attempt": attempt, "temperature": temp, "duplicate": dup})
            if not dup:
                return code, text, msgs, trace, ok, fb
            fault, _ = parse_repair(text)
            note = DUPLICATE_NOTE.format(culprit=culprit(feedback, failed[-1][0], fault))
            msgs = messages + [{"role": "user", "content": error_turn(feedback) + "\n\n" + note}]
        ok, fb = check(code)
        return code, text, msgs, trace, ok, fb  # every resample was a duplicate; trace shows it

    def solve(self, task_messages, entry_point, check, first=None):
        """task_messages: [user turn]; check(code) -> (passed, feedback). first: optional precomputed draft text."""
        text = first if first is not None else self._one(task_messages)
        code = extract_code(text, entry_point)
        ok, fb = check(code)
        log = {"first_pass": ok, "repairs": []}
        msgs = task_messages + [{"role": "assistant", "content": text}]
        failed = [(code, fb)]
        for _ in range(self.max_repairs):
            if ok:
                break
            code, text, used, trace, ok, fb2 = self.repair(msgs, failed, fb, entry_point, check)
            log["repairs"].append({"trace": trace, "passed": ok, "blocked": sum(t["duplicate"] for t in trace),
                                   "still_duplicate": trace[-1]["duplicate"], "text": text})
            msgs = used + [{"role": "assistant", "content": text}]
            failed.append((code, fb2))
            fb = fb2
        log.update(final_pass=ok, code=code)
        return log


if __name__ == "__main__":
    import argparse, json
    import gen, sandbox, tasks
    from schema import ROOT, WORK, jsonl
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--mode", default="tests", choices=["tests", "review"])
    a = ap.parse_args()
    he = {t["id"]: t for t in tasks.humaneval()}
    firsts = {r["id"]: r for r in jsonl(WORK / f"eval_{a.name}.jsonl") if r["part"] == "coding"}
    model, tok = gen.load(a.adapter)
    H = Harness(model, tok, gen.generate)
    res = []
    for i, (tid, r) in enumerate(sorted(firsts.items())):
        t = he[tid]
        if a.mode == "tests":
            check = lambda c, t=t: sandbox.run(t, c)
        else:
            from schema import task_text
            def check(c, t=t):
                ok, out, p = H.self_review(task_text(t), c)
                return ok, out
        if a.mode == "tests" and r["passed"]:
            res.append({"id": tid, "first_pass": True, "final_pass": True, "repairs": []})
            continue
        log = H.solve([{"role": "user", "content": code_prompt(t)}], t["entry_point"], check, first=r["out"])
        log["final_pass_hidden"] = sandbox.run(t, log["code"])[0]
        log["first_pass_hidden"] = r["passed"]
        log["id"] = tid
        res.append(log)
        print(i, tid, {k: log.get(k) for k in ("first_pass", "final_pass")},
              [x["trace"] for x in log["repairs"]], flush=True)
    fp = sum(x.get("first_pass_hidden", x["first_pass"]) for x in res)
    fin = sum(x.get("final_pass_hidden", x["final_pass"]) for x in res)
    reps = [rp for x in res for rp in x["repairs"]]
    summary = {"name": a.name, "mode": a.mode, "n": len(res), "first_pass": fp, "final_pass": fin,
               "repair_passes": len(reps), "repairs_succeeded": sum(rp["passed"] for rp in reps),
               "duplicates_blocked": sum(rp["blocked"] for rp in reps),
               "repairs_still_duplicate": sum(rp["still_duplicate"] for rp in reps)}
    json.dump(summary, open(ROOT / f"results/harness_{a.name}_{a.mode}.json", "w"), indent=1)
    with open(WORK / f"harness_{a.name}_{a.mode}.jsonl", "w") as f:
        for x in res:
            f.write(json.dumps(x) + "\n")
    print(json.dumps(summary, indent=1))
