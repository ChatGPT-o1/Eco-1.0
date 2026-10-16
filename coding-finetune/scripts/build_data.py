"""Build the verified, masked training set (MBPP pool only; HumanEval is held out).

Mix (by example count):
  35% first_pass : prompt -> 1 framing sentence + clean, test-passing code
  45% repair     : [task][broken draft (masked)][real test feedback] -> FAULT / PIVOT / passing fix
                   sources: the base model's own failed drafts (hand-diagnosed in
                   data/annotations/repairs.jsonl), single-bug mutants of passing code, and
                   duplicate-blocker turns in the exact harness format.
  20% review+tools: calibrated FAILURE_PROBABILITY reviews; qwen3_coder tool calls.

Every code target is executed against the task's tests before it is written.
Only the final assistant turn is trained (train.py masks the rest).
"""
import random
import re
import clean, mutate, sandbox, tasks
import tools_data as td
from schema import (DUPLICATE_NOTE, PLACEHOLDER, ROOT, WORK, code_prompt, error_turn, extract_code, filler_hits,
                    jsonl, repair_target, review_prompt, review_target, task_text, tool_message, write_jsonl)
from harness import culprit

rng = random.Random(42)
N_TOTAL = 560
SHARE = {"first_pass": 0.35, "repair": 0.45, "review": 0.10, "tools": 0.10}

pool = {t["id"]: t for t in tasks.mbpp()}
base = {r["id"]: r for r in jsonl(WORK / "base_mbpp_greedy.jsonl")}
import annotations
ann_rep = {r["id"]: r for r in annotations.REPAIRS}
framing = dict(annotations.FRAMING)
stats = {"rejected": {}}


def reject(why):
    stats["rejected"][why] = stats["rejected"].get(why, 0) + 1


def U(c):
    return {"role": "user", "content": c}


def A(c):
    return {"role": "assistant", "content": c}


def voice_ok(text):
    if filler_hits(text):
        reject("filler")
        return False
    return True


VIBE_PREFIX = ["need a fn to", "quick one:", "can you", "write me something to", "gimme a function to", "pls", ""]


def vibe(text):
    t = re.sub(r"^Write (a )?(python )?(function|program|method)( that| which| to)?\s*", "", text, flags=re.I)
    t = t.rstrip(".").strip()
    t = t[0].lower() + t[1:] if t else t
    p = rng.choice(VIBE_PREFIX)
    return f"{p} {t}".strip()


def _sig(code, name):
    m = re.search(rf"^def\s+{re.escape(name)}\s*\(([^)]*)\)", code, re.M)
    return re.sub(r"\s+", "", m.group(1)) if m else None


def stub_from(code, task):
    """HumanEval-style stub whose signature is exactly the target code's (imports + def line + docstring)."""
    m = re.search(rf"^def\s+{re.escape(task['entry_point'])}\s*\(.*?\).*?:\s*$", code, re.M)
    if not m:
        return None
    imports = "".join(l + "\n" for l in code.splitlines() if l.startswith(("import ", "from ")))
    ex = task["tests"][0].replace("assert ", "", 1)
    return f'{imports}{chr(10) if imports else ""}{m.group(0)}\n    """{task["text"]}\n\n    >>> {ex}\n    """\n'


def prompt_for(task, code):
    style = rng.choices(["spec", "signature", "vibe"], [0.45, 0.35, 0.2])[0]
    if style == "signature":
        stub = stub_from(code, task)
        if stub is None or "class " in code:
            style = "spec"
        else:
            task = dict(task, stub=stub)
    if style == "vibe":
        task = dict(task, vibe=vibe(task["text"]))
    return code_prompt(task, style)


def err_turn(fb):
    if rng.random() < 0.8:
        return error_turn(fb)  # exact harness/eval format
    lead = rng.choice(["still failing:", "hmm, got this:", "ran it:", "nope"])
    return f"{lead}\n```\n{fb.strip()}\n```"


def fenced(code):
    return f"```python\n{code.strip()}\n```"


# ---------------------------------------------------------------- verified solutions
solutions = {}  # id -> clean passing code (on-policy where possible)
for tid, r in base.items():
    if r["passed"]:
        c = clean.clean_code(r["code"])
        ok, _ = sandbox.run(pool[tid], c)
        solutions[tid] = c if ok else r["code"]

fixed = {}  # id -> (fault, pivot, fixed clean code) from hand-written repairs of the base's own drafts
for tid, a in ann_rep.items():
    draft = base[tid]["code"]
    if "code" in a:
        new = a["code"]
    else:
        new = draft
        for old, rep in a["edits"]:
            if new.count(old) != 1:
                reject("edit_not_applicable")
                new = None
                break
            new = new.replace(old, rep, 1)
    if new is None:
        continue
    new = clean.clean_code(new)
    ok, fb = sandbox.run(pool[tid], new)
    if not ok:
        reject("annotated_fix_fails")
        print("ANNOTATED FIX FAILS", tid, fb[:200])
        continue
    fixed[tid] = (a["fault"], a["pivot"], new)

print(f"base passing {len(solutions)}, hand-fixed failures {len(fixed)}")

# ---------------------------------------------------------------- first pass
first = []
for tid in sorted(set(solutions) | set(fixed)):
    if tid not in framing:
        continue
    code = annotations.FIRST_CODE.get(tid) or solutions.get(tid) or fixed[tid][2]
    if tid in annotations.FIRST_CODE and not sandbox.run(pool[tid], code)[0]:
        reject("first_code_override_fails")
        continue
    if PLACEHOLDER.search(code):
        reject("placeholder")
        continue
    target = f"{framing[tid]}\n\n{fenced(code)}"
    if not voice_ok(target):
        continue
    first.append({"kind": "first_pass", "id": tid, "messages": [U(prompt_for(pool[tid], code)), A(target)]})

# ---------------------------------------------------------------- repair: on-policy drafts
rep_onpolicy = []
for tid, (fault, pivot, code) in fixed.items():
    t = pool[tid]
    target = repair_target(fault, pivot, code)
    if not voice_ok(target):
        continue
    rep_onpolicy.append({"kind": "repair_onpolicy", "id": tid, "messages": [
        U(code_prompt(t)), A(base[tid]["response"]), U(err_turn(base[tid]["feedback"])), A(target)]})

# ---------------------------------------------------------------- repair: mutants of passing code
rep_mut, rep_dup, review_bug = [], [], []
sources = {tid: code for tid, code in solutions.items()}
sources.update({tid: f[2] for tid, f in fixed.items()})
mut_ids = sorted(sources)
rng.shuffle(mut_ids)
op_count = {}
for tid in mut_ids:
    t, src = pool[tid], sources[tid]
    ms = mutate.mutants(src, seed=hash(tid) % 1000)
    ms.sort(key=lambda m: op_count.get(m["op"], 0))  # balance bug types
    used_ops = set()
    for m in ms[:8]:
        if len(used_ops) == 2:
            break
        if m["op"] in used_ops:
            continue
        ok, fb = sandbox.run(t, m["code"])
        if ok or fb.startswith("TimeoutError"):
            continue
        used_ops.add(m["op"])
        op_count[m["op"]] = op_count.get(m["op"], 0) + 1
        lead = framing.get(tid, "Straightforward version:")
        draft = f"{lead}\n\n{fenced(m['code'])}"
        target = repair_target(m["fault"], m["pivot"], src)
        p = U(code_prompt(t))
        r = rng.random()
        if r < 0.10:  # duplicate-blocker turn, exactly as harness.py builds it
            note = DUPLICATE_NOTE.format(culprit=culprit(fb, m["code"]))
            rep_dup.append({"kind": "repair_duplicate", "id": tid, "messages": [
                p, A(draft), U(error_turn(fb) + "\n\n" + note), A(target)]})
        elif r < 0.30:
            review_bug.append({"kind": "review_bug", "id": tid, "messages": [
                U(review_prompt(task_text(t), m["code"])),
                A(review_target(m["fault"], rng.choice([80, 85, 90, 95])))]})
        else:
            rep_mut.append({"kind": "repair_mutant", "id": tid, "messages": [p, A(draft), U(err_turn(fb)), A(target)]})

# ---------------------------------------------------------------- review: on-policy buggy + passing
for tid, (fault, pivot, code) in fixed.items():
    try:
        compile(base[tid]["code"], "draft", "exec")
    except SyntaxError:
        continue  # prose-only / truncated drafts are not reviewable code
    review_bug.append({"kind": "review_bug", "id": tid, "messages": [
        U(review_prompt(task_text(pool[tid]), base[tid]["code"])),
        A(review_target(fault, rng.choice([75, 80, 85, 90])))]})
review_ok = []
OK_LINES = ["Traced `{call}` and got `{exp}`, which matches. Nothing fragile stands out.",
            "`{call}` returns `{exp}` as required, and I don't see an input it would mishandle.",
            "Checked `{call}` → `{exp}`; correct. Low risk."]
for tid in sorted(solutions):
    t = pool[tid]
    m = re.match(r"assert\s+(.+?)\s*==\s*(.+)$", t["tests"][-1])
    if not m or len(m.group(0)) > 160:
        continue
    finding = rng.choice(OK_LINES).format(call=m.group(1).strip(), exp=m.group(2).strip())
    review_ok.append({"kind": "review_ok", "id": tid, "messages": [
        U(review_prompt(task_text(t), solutions[tid])), A(review_target(finding, rng.choice([5, 10, 15, 20])))]})

# ---------------------------------------------------------------- tools
tools = []
for sc in td.TRAIN:
    if "target" in sc:
        th, name, args = sc["target"]
        last = tool_message(name, args, th)
    else:
        last = A(sc["answer"])
    tools.append({"kind": "tool", "tools": sc["tools"], "messages": sc["messages"] + [last]})

FILES = ["src/app.py", "lib/db.py", "README.md", "tests/test_api.py", "docs/setup.md", "server/main.go",
         "web/index.ts", "config/prod.yaml", "scripts/deploy.sh", "pkg/cache/lru.py"]
KEYS = ["cache", "login", "invoice", "retry", "parser", "upload", "search", "export"]
CITIES = ["Nairobi", "Osaka", "Bogotá", "Tallinn", "Perth", "Quito"]
for i in range(30):
    k = rng.randrange(5)
    if k == 0:
        f = rng.choice(FILES)
        msg, tgt = rng.choice(["show me {f}", "open {f}", "what's in {f}?", "cat {f}"]).format(f=f), \
            (rng.choice(["Opening it.", "Reading it.", "Pulling it up."]), "read_file", {"path": f})
        tl = td.DEV
    elif k == 1:
        kw = rng.choice(KEYS)
        msg, tgt = f"run the {kw} tests", (f"Running only the {kw} tests.", "run_tests", {"keyword": kw})
        tl = td.DEV
    elif k == 2:
        kw = rng.choice(KEYS)
        msg, tgt = f"where is {kw}_handler defined?", ("Searching for the definition.", "search_code",
                                                      {"pattern": f"def {kw}_handler"})
        tl = td.DEV
    elif k == 3:
        c = rng.choice(CITIES)
        u = rng.choice(["metric", "imperial"])
        msg, tgt = f"weather in {c}, {u} units", ("Checking.", "get_weather", {"city": c, "units": u})
        tl = td.T("get_weather", "send_message")
    else:
        n = rng.choice([5, 10, 15, 45, 90])
        msg, tgt = f"set a {n} minute timer for tea", ("Starting it.", "set_timer", {"minutes": n, "label": "tea"})
        tl = td.T("set_timer", "get_weather")
    tools.append({"kind": "tool", "tools": tl, "messages": [U(msg), tool_message(tgt[1], tgt[2], tgt[0])]})

# ---------------------------------------------------------------- mix
def take(rows, n):
    rows = rows[:]
    rng.shuffle(rows)
    return rows[:n]


n_first = round(N_TOTAL * SHARE["first_pass"])
n_rep = round(N_TOTAL * SHARE["repair"])
n_rev = round(N_TOTAL * SHARE["review"])
n_tool = N_TOTAL - n_first - n_rep - n_rev
repairs = rep_onpolicy + rep_dup + take(rep_mut, max(0, n_rep - len(rep_onpolicy) - len(rep_dup)))
half = n_rev // 2
reviews = take(review_bug, half) + take(review_ok, n_rev - min(half, len(review_bug)))
toolrows = take(tools, n_tool)
data = take(first, n_first) + repairs[:n_rep] + reviews + toolrows
rng.shuffle(data)
counts = {}
for r in data:
    counts[r["kind"]] = counts.get(r["kind"], 0) + 1
print("available:", {"first": len(first), "rep_onpolicy": len(rep_onpolicy), "rep_mut": len(rep_mut),
                     "rep_dup": len(rep_dup), "review_bug": len(review_bug), "review_ok": len(review_ok),
                     "tools": len(tools)})
print("mix:", counts, "total", len(data), "ops:", op_count, "rejected:", stats["rejected"])
groups = {"first_pass": ["first_pass"], "repair": ["repair_onpolicy", "repair_mutant", "repair_duplicate"],
          "review+tools": ["review_bug", "review_ok", "tool"]}
print("shares:", {g: round(sum(counts.get(k, 0) for k in ks) / len(data), 3) for g, ks in groups.items()})
write_jsonl(ROOT / "data/train.jsonl", data)
