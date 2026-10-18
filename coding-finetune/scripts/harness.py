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


def repair_batch(model, tok, generate, cases, resamples=3, batch=16, log=print):
    """Batched form of Harness.repair for evaluation (same messages, same duplicate rules).

    cases: list of dicts with messages (task + failed draft), failed [(code, feedback)], feedback,
           entry_point, check. Returns per-case {code, text, passed, trace}.
    """
    out = [None] * len(cases)
    pending = list(range(len(cases)))
    msgs = {i: c["messages"] + [{"role": "user", "content": error_turn(c["feedback"])}] for i, c in enumerate(cases)}
    traces = {i: [] for i in range(len(cases))}
    for attempt in range(resamples + 1):
        if not pending:
            break
        temp = 0.0 if attempt == 0 else min(0.4 + 0.2 * attempt, 1.0)
        texts = generate(model, tok, [{"messages": msgs[i]} for i in pending], max_new=1024, batch=batch,
                         temperature=temp, seed=attempt)
        still = []
        for i, (text, _) in zip(pending, texts):
            c = cases[i]
            code = extract_code(text, c["entry_point"])
            dup = any(is_duplicate(code, f, c["entry_point"]) for f, _ in c["failed"])
            ok, fb = (False, None) if dup else c["check"](code)
            if not dup and not ok:
                dup = any(fb == ffb and similarity(code, f, c["entry_point"]) >= NO_PROGRESS_SIM for f, ffb in c["failed"])
            traces[i].append({"attempt": attempt, "temperature": temp, "duplicate": dup})
            out[i] = {"code": code, "text": text, "passed": ok, "feedback": fb, "trace": traces[i]}
            if dup and attempt < resamples:
                fault, _ = parse_repair(text)
                note = DUPLICATE_NOTE.format(culprit=culprit(c["feedback"], c["failed"][-1][0], fault))
                msgs[i] = c["messages"] + [{"role": "user", "content": error_turn(c["feedback"]) + "\n\n" + note}]
                still.append(i)
        log(f"attempt {attempt}: {len(pending)} generated, {len(still)} duplicates re-sampled")
        pending = still
    return out


if __name__ == "__main__":
    import argparse, json
    import gen, sandbox, tasks
    from schema import ROOT, WORK, jsonl
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="evaluate.py run whose first-pass outputs to repair")
    ap.add_argument("--adapter")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or a.name
    he = {t["id"]: t for t in tasks.humaneval()}
    firsts = {r["id"]: r for r in jsonl(WORK / f"eval_{a.name}.jsonl") if r["part"] == "coding"}
    model, tok = gen.load(a.adapter)
    cases, ids = [], []
    for tid, r in sorted(firsts.items()):
        if r["passed"]:
            continue
        t = he[tid]
        code = extract_code(r["out"], t["entry_point"])
        cases.append({"messages": [{"role": "user", "content": code_prompt(t)}, {"role": "assistant", "content": r["out"]}],
                      "failed": [(code, r["feedback"])], "feedback": r["feedback"], "entry_point": t["entry_point"],
                      "check": (lambda c, t=t: sandbox.run(t, c))})
        ids.append(tid)
    res = repair_batch(model, tok, gen.generate, cases)
    n = len(firsts)
    fp = sum(r["passed"] for r in firsts.values())
    fixed = sum(r["passed"] for r in res)
    summary = {"name": tag, "n": n, "first_pass": fp, "repair_attempts": len(res), "repairs_succeeded": fixed,
               "final_pass": fp + fixed,
               "duplicates_blocked": sum(sum(t["duplicate"] for t in r["trace"]) for r in res),
               "cases_with_a_blocked_duplicate": sum(any(t["duplicate"] for t in r["trace"]) for r in res),
               "still_duplicate_after_resamples": sum(r["trace"][-1]["duplicate"] for r in res),
               "fault_pivot_format": sum(all(parse_repair(r["text"])) for r in res) / max(len(res), 1)}
    json.dump(summary, open(ROOT / f"results/harness_{tag}.json", "w"), indent=1)
    with open(WORK / f"harness_{tag}.jsonl", "w") as f:
        for tid, r in zip(ids, res):
            f.write(json.dumps({"id": tid, **r}) + "\n")
    print(json.dumps(summary, indent=1))
