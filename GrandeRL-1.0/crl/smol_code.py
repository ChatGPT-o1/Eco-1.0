#!/usr/bin/env python3
"""Make SmolLM2-135M-Instruct better at Python with otnp_v6: no gradient training, only 20 step sizes chosen under KL budgets.

  data     100 MBPP train problems: 4 sampled answers + the reference solution (the "edit") each, rated 1-10 by running the
           problem's tests; 40 ordinary chat replies whose behaviour should be preserved
  dirs     20 weight directions = the gradient of the reference solutions' log-likelihood on 150 OTHER MBPP train problems,
           restricted to the norm gains (n) and down_proj (d) of 10 groups of 3 layers
  measure  dz.pt: the logits at 33 stored candidates per token and their change for a unit step along each direction
  eval     pass@1, greedy, on HumanEval (164) and the first 200 MBPP test problems, for one model folder

Ratings: 1 = no function or a syntax error, 2 = crashes on load or times out, 3 + 7 * (share of tests passed) otherwise.

  python -I smol_code.py data    --model DIR --out W/code
  python -I smol_code.py dirs    --model DIR --out W/code
  python -I smol_code.py measure --model DIR --out W/code
  python -I otnp_v6.py --measure W/code/dz.pt --scored W/code/scored.jsonl --preserve W/code/preserve_flagged.jsonl \\
                       --dirs W/code/dirs.pt --out W/code/run2 --model DIR
  python -I smol_code.py eval    --model DIR --out W/code --tag base
"""
import argparse, json, os, re, resource, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # evaluate.py sits next to this file
from evaluate import HE_INSTR, build_program, extract_code, run_program  # noqa: E402

torch.set_num_threads(min(4, os.cpu_count() or 4))
MBPP_INSTR = "{text}\nYour code should pass this test:\n{test}\nWrite the Python function in a single ```python code block."
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def load(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    return tok, model


def chat_ids(tok, user):
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def mbpp(split):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download("google-research-datasets/mbpp", f"full/{split}-00000-of-00001.parquet", repo_type="dataset")
    return pq.read_table(p).to_pylist()


def eos_ids(tok):
    return sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None})


def generate(model, tok, prompts, max_new, sample, n_ret=1, batch=16, temperature=0.8, top_p=0.95):
    """Batched generation (left padding). Returns, per prompt, n_ret lists of new token ids cut after the first end token,
    and whether each one was cut by the length limit."""
    eos = eos_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    res = [None] * len(prompts)
    for s in range(0, len(order), batch):
        grp = order[s:s + batch]
        L = max(len(prompts[i]) for i in grp)
        ids = torch.full((len(grp), L), pad, dtype=torch.long)
        att = torch.zeros(len(grp), L, dtype=torch.long)
        for j, i in enumerate(grp):
            ids[j, L - len(prompts[i]):] = torch.tensor(prompts[i])
            att[j, L - len(prompts[i]):] = 1
        kw = dict(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n_ret) if sample else dict(do_sample=False)
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, eos_token_id=eos, pad_token_id=pad, **kw)
        gen = out[:, L:].tolist()
        for j, i in enumerate(grp):
            rows = []
            for row in gen[j * n_ret:(j + 1) * n_ret]:
                cut = next((k for k, t in enumerate(row) if t in eos), None)
                rows.append((row if cut is None else row[:cut + 1], cut is None))
            res[i] = rows
        log(f"  generated {min(s + batch, len(order))}/{len(order)} prompts")
    return res


HARNESS = '''
try:
    compile(src, "cand", "exec")
except Exception:
    print("@@RESULT@@ syntax"); raise SystemExit(0)
g = {"__name__": "cand"}
try:
    exec(setup + "\\n" + src, g)
except BaseException:
    print("@@RESULT@@ crash"); raise SystemExit(0)
n = 0
for t in tests:
    try:
        exec(t, g)
        n += 1
    except BaseException:
        pass
print("@@RESULT@@", n, len(tests))
'''


def run_tests(code, tests, setup, timeout=10):
    """(tests passed, total) or 'syntax' / 'crash', running the code in a separate, memory-limited Python."""
    prog = f"src = {code!r}\ntests = {tests!r}\nsetup = {setup!r}\n" + HARNESS

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    with tempfile.TemporaryDirectory(prefix="mbpp_") as d:
        path = os.path.join(d, "prog.py")
        open(path, "w").write(prog)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, errors="replace",
                               timeout=timeout, preexec_fn=limits, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return "crash"
    m = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")]
    if not m:
        return "crash"
    f = m[-1].split()[1:]
    return f[0] if len(f) == 1 else (int(f[0]), int(f[1]))


def rate(text, prob):
    code = extract_code(text)
    if not re.search(r"^\s*def\s+\w+\s*\(", code, re.M):
        return 1, "no function"
    out = run_tests(code, prob["test_list"], prob["test_setup_code"] or "")
    if out == "syntax":
        return 1, "syntax"
    if out == "crash":
        return 2, "crash"
    return 3 + round(7 * out[0] / out[1]), f"{out[0]}/{out[1]}"


def ref_reply(prob):
    return "```python\n" + prob["code"].replace("\r\n", "\n").strip() + "\n```"


# ------------------------------------------------------------------------------------------------- stages
def stage_data(a):
    tok, model = load(a.model)
    torch.manual_seed(a.seed)
    train = mbpp("train")[:a.n_prompts]
    prompts = [chat_ids(tok, MBPP_INSTR.format(text=p["text"], test=p["test_list"][0])) for p in train]
    log(f"sampling 4 answers for {len(train)} MBPP train problems")
    samples = generate(model, tok, prompts, a.max_new, sample=True, n_ret=4, batch=a.batch)
    end = eos_ids(tok)[0]
    seq_ids, rs, recs, jobs = [], [], [], []
    for p, pr, outs in zip(train, prompts, samples):
        cands = []
        for toks, trunc in outs:
            cands.append(dict(source="sample", trunc=trunc, text=tok.decode(toks, skip_special_tokens=True), toks=toks))
        rtoks = tok(ref_reply(p), add_special_tokens=False)["input_ids"] + [end]
        cands.append(dict(source="edit", trunc=False, text=ref_reply(p), toks=rtoks))
        for c in cands:
            seq_ids.append(pr + c.pop("toks")); rs.append(len(pr))
            jobs.append((c, p))
        recs.append(dict(id=len(recs), task_id=p["task_id"], prompt=MBPP_INSTR.format(text=p["text"], test=p["test_list"][0]), cands=cands))
    log(f"running the tests for {len(jobs)} answers")
    with ThreadPoolExecutor(4) as ex:
        rated = list(ex.map(lambda j: rate(j[0]["text"], j[1]), jobs))
    for (c, _), (r, why) in zip(jobs, rated):
        c["r"], c["why"] = r, why
    held = [json.loads(l) for l in open(a.chat)][:200]
    chat = [m["messages"][0]["content"] for m in held if m["messages"][0]["role"] == "user" and len(m["messages"][0]["content"]) < 600][:a.n_pres]
    cp = [chat_ids(tok, q) for q in chat]
    log(f"greedy replies for {len(cp)} chat prompts to preserve")
    pres = generate(model, tok, cp, 64, sample=False, batch=a.batch)
    with open(os.path.join(a.out, "preserve_flagged.jsonl"), "w") as f:
        for q, pr, ((toks, _),) in zip(chat, cp, pres):
            seq_ids.append(pr + toks); rs.append(len(pr))
            f.write(json.dumps(dict(prompt=q, reply=tok.decode(toks, skip_special_tokens=True))) + "\n")
    with open(os.path.join(a.out, "scored.jsonl"), "w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    torch.save(dict(seq_ids=[torch.tensor(x) for x in seq_ids], resp_start=rs, n_scored=5 * len(recs)), os.path.join(a.out, "seqs.pt"))
    sr = [c["r"] for rec in recs for c in rec["cands"] if c["source"] == "sample"]
    er = [c["r"] for rec in recs for c in rec["cands"] if c["source"] == "edit"]
    hist = {r: sr.count(r) for r in sorted(set(sr))}
    log(f"sample ratings {hist}; samples passing every test {sum(r == 10 for r in sr)}/{len(sr)}; "
        f"problems with at least one passing sample {sum(any(c['r'] == 10 for c in rec['cands'][:4]) for rec in recs)}/{len(recs)}; "
        f"reference solutions rated 10: {sum(r == 10 for r in er)}/{len(er)}")


def group_params(model, n_groups=10):
    nl = model.config.num_hidden_layers
    per = nl // n_groups
    groups = []
    for g in range(n_groups):
        ls = range(g * per, (g + 1) * per)
        groups.append(([f"model.layers.{l}.{n}.weight" for l in ls for n in ("input_layernorm", "post_attention_layernorm")],
                       [f"model.layers.{l}.mlp.down_proj.weight" for l in ls]))
    return groups


def stage_dirs(a):
    tok, model = load(a.model)
    probs = mbpp("train")[a.n_prompts:a.n_prompts + a.n_dir]
    params = dict(model.named_parameters())
    groups = group_params(model)
    wanted = {pn for gn, gd in groups for pn in gn + gd}
    for pn, p in params.items():
        p.requires_grad_(pn in wanted)
    end = eos_ids(tok)[0]
    seqs = []
    for p in probs:
        pr = chat_ids(tok, MBPP_INSTR.format(text=p["text"], test=p["test_list"][0]))
        rep = tok(ref_reply(p), add_special_tokens=False)["input_ids"] + [end]
        seqs.append((pr, rep))
    n_tok = sum(len(r) for _, r in seqs)
    log(f"gradient of the reference solutions' log-likelihood on {len(seqs)} other MBPP problems ({n_tok} tokens)")
    pad = tok.pad_token_id
    for s in range(0, len(seqs), 8):
        grp = seqs[s:s + 8]
        L = max(len(p) + len(r) for p, r in grp)
        ids = torch.full((len(grp), L), pad, dtype=torch.long)
        lab = torch.full((len(grp), L), -100, dtype=torch.long)
        att = torch.zeros(len(grp), L, dtype=torch.long)
        for j, (p, r) in enumerate(grp):
            x = p + r
            ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
            lab[j, len(p):len(x)] = torch.tensor(r)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        (nll / n_tok).backward()
    names, dirs = [], {}
    for g, (gn, gd) in enumerate(groups):
        for kind, pns in (("n", gn), ("d", gd)):
            d = {pn: -params[pn].grad.detach().clone() for pn in pns}          # uphill on the log-likelihood
            norm = sum(float(t.pow(2).sum()) for t in d.values()) ** 0.5
            dirs[f"{kind}{g}"] = {pn: t / norm for pn, t in d.items()}
            names.append(f"{kind}{g}")
            log(f"  {kind}{g}: gradient norm {norm:.3e}")
    torch.save(dict(names=names, dirs=dirs), os.path.join(a.out, "dirs.pt"))
    log(f"saved {len(names)} directions")


def hidden_rows(model, ids_list, rs, seqs, batch):
    """Final hidden states at the reply positions (the rows that predict each reply token), batched with right padding."""
    out = {}
    order = sorted(seqs, key=lambda y: len(ids_list[y]))
    for s in range(0, len(order), batch):
        grp = order[s:s + batch]
        L = max(len(ids_list[y]) for y in grp)
        ids = torch.zeros(len(grp), L, dtype=torch.long)
        att = torch.zeros(len(grp), L, dtype=torch.long)
        for j, y in enumerate(grp):
            ids[j, :len(ids_list[y])] = ids_list[y]; att[j, :len(ids_list[y])] = 1
        with torch.inference_mode():
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
        for j, y in enumerate(grp):
            out[y] = h[j, rs[y] - 1: len(ids_list[y]) - 1].clone()
    return out


def stage_measure(a):
    tok, model = load(a.model)
    sq = torch.load(os.path.join(a.out, "seqs.pt"), weights_only=False)
    ids_list, rs, n_scored = sq["seq_ids"], sq["resp_start"], sq["n_scored"]
    dd = torch.load(os.path.join(a.out, "dirs.pt"), weights_only=False)
    names, dirs = dd["names"], dd["dirs"]
    params = dict(model.named_parameters())
    W = model.get_output_embeddings().weight.detach()             # float32 is plenty for logit changes of ~0.15
    n_seq = len(ids_list)
    allseq = list(range(n_seq))
    log(f"base pass over {n_seq} sequences")
    H = hidden_rows(model, ids_list, rs, allseq, a.batch)
    g = torch.Generator().manual_seed(a.seed)
    idx_l, obs_l, seq_l, z0_l = [], [], [], []
    for y in allseq:
        h = H[y]
        full = h @ W.T
        obs = ids_list[y][rs[y]:]
        masked = full.clone()
        masked[torch.arange(len(obs)), obs] = -float("inf")
        cand = torch.cat([obs[:, None], masked.topk(32, dim=1).indices], 1)
        perm = torch.stack([torch.randperm(33, generator=g) for _ in range(len(obs))])
        cand = cand.gather(1, perm)
        idx_l.append(cand); obs_l.append(obs); seq_l.append(torch.full((len(obs),), y)); z0_l.append(full.gather(1, cand))
    idx, obs, seq, z0 = torch.cat(idx_l), torch.cat(obs_l), torch.cat(seq_l), torch.cat(z0_l).double()
    N = len(seq)
    offs = torch.cumsum(torch.tensor([0] + [len(o) for o in obs_l]), 0)
    log(f"{N} tokens")

    def z_at(Hd, ys):
        return {y: torch.einsum("nd,nkd->nk", Hd[y], W[idx[offs[y]:offs[y + 1]]]).double() for y in ys}

    def response(name, scale, ys):
        backup = {pn: params[pn].detach().clone() for pn in dirs[name]}
        with torch.no_grad():
            for pn, d in dirs[name].items():
                params[pn].add_(d, alpha=scale)
        Hd = hidden_rows(model, ids_list, rs, ys, a.batch)
        with torch.no_grad():
            for pn in dirs[name]:
                params[pn].copy_(backup[pn])
        zd = z_at(Hd, ys)
        return {y: zd[y] - z0[offs[y]:offs[y + 1]] for y in ys}

    probe_seqs = allseq[::max(1, n_seq // 40)]
    dz = torch.empty(N, 33, len(names), dtype=torch.float64)
    scale = []
    for i, name in enumerate(names):
        step = 1e-2
        pr = response(name, step, probe_seqs)
        rms = float(torch.cat([v.reshape(-1) for v in pr.values()]).pow(2).mean().sqrt())
        s = step * a.target_rms / rms
        full = response(name, s, allseq)
        for y in allseq:
            dz[offs[y]:offs[y + 1], :, i] = full[y]
        scale.append(s)
        log(f"  {name}: scale {s:.3e}, unit-step rms {float(dz[:, :, i].pow(2).mean().sqrt()):.3f}")
    sid = torch.arange(n_seq)
    is_edit = torch.zeros(n_seq, dtype=torch.bool)
    is_edit[4:n_scored:5] = True
    torch.save(dict(idx=idx, z0=z0.float(), obs=obs, p0=torch.softmax(z0, 1).float(), seq=seq,
                    task=(seq < n_scored) & ~is_edit[seq], pres=seq >= n_scored, scale=torch.tensor(scale), names=names,
                    dz=dz.float(), seq_ids=ids_list, resp_start=rs), os.path.join(a.out, "dz.pt"))
    log(f"saved dz.pt ({N} tokens x 33 x {len(names)})")


def stage_eval(a):
    tok, model = load(a.model)
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    he = pq.read_table(hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet",
                                       repo_type="dataset")).to_pylist()[:a.n_he]
    mt = mbpp("test")[:a.n_test]
    res = {}
    log(f"HumanEval: {len(he)} problems")
    outs = generate(model, tok, [chat_ids(tok, HE_INSTR.format(prompt=p["prompt"])) for p in he], 256, sample=False, batch=a.batch)
    per = []
    for p, ((toks, _),) in zip(he, outs):
        text = tok.decode(toks, skip_special_tokens=True)
        ok, why = run_program(build_program(p["prompt"], p["entry_point"], extract_code(text), p["test"]))
        per.append(dict(task_id=p["task_id"], passed=ok))
    res["humaneval"] = dict(passed=sum(x["passed"] for x in per), total=len(per), per=per)
    log(f"HumanEval {res['humaneval']['passed']}/{len(per)}")
    log(f"MBPP test: {len(mt)} problems")
    outs = generate(model, tok, [chat_ids(tok, MBPP_INSTR.format(text=p["text"], test=p["test_list"][0])) for p in mt], 192,
                    sample=False, batch=a.batch)
    with ThreadPoolExecutor(4) as ex:
        rated = list(ex.map(lambda j: rate(tok.decode(j[1][0][0], skip_special_tokens=True), j[0]), zip(mt, outs)))
    per = [dict(task_id=p["task_id"], passed=r == 10, r=r) for p, (r, _) in zip(mt, rated)]
    res["mbpp_test"] = dict(passed=sum(x["passed"] for x in per), total=len(per), per=per,
                            mean_rating=sum(x["r"] for x in per) / len(per))
    log(f"MBPP test {res['mbpp_test']['passed']}/{len(per)} (mean rating {res['mbpp_test']['mean_rating']:.2f})")
    json.dump(res, open(os.path.join(a.out, f"eval_{a.tag}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["data", "dirs", "measure", "eval"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--n-dir", type=int, default=150)
    ap.add_argument("--n-pres", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-he", type=int, default=164)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--target-rms", type=float, default=0.15)
    ap.add_argument("--chat", default="/tmp/claude-0/-home-user-Eco-1-0/65d1f45a-5eb7-5658-9b21-09458b831e94/scratchpad/local/msdata/heldout_prompts.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dict(data=stage_data, dirs=stage_dirs, measure=stage_measure, eval=stage_eval)[a.stage](a)


if __name__ == "__main__":
    main()
