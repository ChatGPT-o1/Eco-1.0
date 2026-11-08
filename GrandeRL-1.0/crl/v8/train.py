#!/usr/bin/env python3
"""Plain RL (GRPO) and V8 on SmolLM2-135M-Instruct, rewarded by MBPP tests. Same engine, same problems, same sampler, same
compute clock; only the update differs.

Every round: draw problems (weighted by the chance that a group of answers is mixed pass/fail), sample --group answers per
problem at T=1, grade them (1 = passes every test), and update.

  grpo  one AdamW step on the group-normalised policy gradient + beta * k3 KL to the starting model (fixed --lr).
  v8    * more problems are drawn until the round holds --min-inf mixed groups (up to --max-prompts);
        * problems where every answer failed get a known-good answer: the model's own earlier test-passing answer if it has
          one, else a test-passing solution of Qwen2.5-Coder-0.5B (bank.py); weight --rescue-weight;
        * the round is split into two halves by problem; the gradient of each half is scaled per weight (Adam-style second
          moment) and the step is shrunk by rho = the share of the direction both halves agree on (rho <= 0: no step);
        * backtracking line search on the REAL model: try step s, s/2, s/4, ... and keep the first that improves the
          objective while KL(old -> new) on the round's answers <= --eps-task and KL(base -> new) on ordinary chat replies
          <= --eps-pres (a hard cap on total drift); no learning rate to set;
        * up to --epochs passes over the same answers (PPO-clipped importance ratios), all inside the same KL budget;
        * every --dev-every rounds the model is scored on MBPP validation; it rolls back to the best weights if the score
          falls more than 2 standard errors below the best.
Both methods keep the best-on-validation weights as their final model.

  python -I train.py --method grpo --out DIR --budget-min 240
  python -I train.py --method v8   --out DIR --budget-min 240
  python -I train.py --selfcheck
"""
import argparse, json, math, os, random, resource, sys, time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

DATA = os.path.join(HERE, "data")
CHAT = f"{E.SCR}/local/msdata/heldout_prompts.jsonl"


# ------------------------------------------------------------------------------------------------ sampler
def fit_prior(passes, k):
    """Beta(a0, b0) prior over the per-problem pass rate, fitted to the screening counts by maximum likelihood
    (beta-binomial, grid search). Most problems are never solved, so the prior is strongly skewed towards 0."""
    best = None
    for a0 in [0.02 * 1.25 ** i for i in range(30)]:
        for b0 in [0.1 * 1.25 ** i for i in range(40)]:
            ll = 0.0
            for x in passes:
                ll += (math.lgamma(a0 + x) + math.lgamma(b0 + k - x) - math.lgamma(a0 + b0 + k)
                       - math.lgamma(a0) - math.lgamma(b0) + math.lgamma(a0 + b0))
            if best is None or ll > best[0]:
                best = (ll, a0, b0)
    return best[1], best[2]


class Sampler:
    """Beta(1 + passes, 1 + fails) per problem; weight = probability that a group of G answers is mixed."""

    def __init__(self, probs, screen, G):
        self.probs = probs
        self.G = G
        k = screen["k"]
        a0, b0 = fit_prior([screen["passes"][str(p["task_id"])] for p in probs], k)
        self.prior = (a0, b0)
        self.a = {p["task_id"]: a0 + screen["passes"][str(p["task_id"])] for p in probs}
        self.b = {p["task_id"]: b0 + k - screen["passes"][str(p["task_id"])] for p in probs}

    def share_never_solved(self):
        """Share of draws that go to problems with no pass so far (for the log)."""
        w = {p["task_id"]: self.weight(p["task_id"]) for p in self.probs}
        z = [t for t in w if self.a[t] - self.prior[0] < 0.5]
        return sum(w[t] for t in z) / sum(w.values())

    def weight(self, tid):
        a, b, G = self.a[tid], self.b[tid], self.G
        ep = eq = 1.0
        for j in range(G):
            ep *= (a + j) / (a + b + j)
            eq *= (b + j) / (a + b + j)
        return max(1e-3, 1.0 - ep - eq)

    def draw(self, n, rng, exclude=()):
        pool = [p for p in self.probs if p["task_id"] not in exclude]
        w = [self.weight(p["task_id"]) for p in pool]
        out = []
        for _ in range(min(n, len(pool))):
            i = rng.choices(range(len(pool)), weights=w)[0]
            out.append(pool.pop(i))
            w.pop(i)
        return out

    def update(self, tid, passes, total):
        self.a[tid] += passes
        self.b[tid] += total - passes

    def state(self):
        return dict(a=self.a, b=self.b)

    def load(self, st):
        self.a, self.b = {int(k): v for k, v in st["a"].items()}, {int(k): v for k, v in st["b"].items()}


# ------------------------------------------------------------------------------------------------ preserve set
def build_preserve(tok, base, n=24, reply=64, top=512):
    """Greedy replies of the starting model to ordinary chat prompts, with its top-`top` log-probs at every reply token."""
    held = [json.loads(l) for l in open(CHAT)][:300]
    qs = [m["messages"][0]["content"] for m in held if m["messages"][0]["role"] == "user" and len(m["messages"][0]["content"]) < 500][:n]
    pr = [E.chat_ids(tok, q) for q in qs]
    outs = E.sample(base, tok, pr, 1, reply, temperature=0, stop_fence=False)
    seqs, plens = [p + o[0] for p, o in zip(pr, outs)], [len(p) for p in pr]
    with torch.inference_mode():
        lps = full_logprobs(base, seqs, plens)
    idx, val = [], []
    for lp in lps:
        v, i = lp.topk(top, -1)
        idx.append(i)
        val.append(v)
    return dict(seqs=seqs, plens=plens, idx=idx, val=val)


def full_logprobs(model, seqs, plens, micro=8):
    res = []
    pad = 0
    for s in range(0, len(seqs), micro):
        mb, mp = seqs[s:s + micro], plens[s:s + micro]
        L = max(len(x) for x in mb)
        ids = torch.full((len(mb), L), pad, dtype=torch.long)
        att = torch.zeros((len(mb), L), dtype=torch.long)
        for j, x in enumerate(mb):
            ids[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        lg = model(input_ids=ids, attention_mask=att).logits.float()
        for j, (x, p) in enumerate(zip(mb, mp)):
            res.append(torch.log_softmax(lg[j, p - 1:len(x) - 1], -1))
    return res


@torch.inference_mode()
def preserve_kl(model, pres):
    """Mean over reply tokens of KL(base || model), with the base distribution coarsened to its top tokens + the rest."""
    lps = full_logprobs(model, pres["seqs"], pres["plens"])
    tot = n = 0.0
    for lp, i, v in zip(lps, pres["idx"], pres["val"]):
        pb = v.exp()
        ln = lp.gather(1, i)
        rb = (1 - pb.sum(1)).clamp(min=1e-12)
        rn = (1 - ln.exp().sum(1)).clamp(min=1e-12)
        kl = (pb * (v - ln)).sum(1) + rb * (rb.log() - rn.log())
        tot += float(kl.sum())
        n += len(kl)
    return tot / n


# ------------------------------------------------------------------------------------------------ helpers
def flat_lp(model, seqs, plens, pad, autocast):
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        return E.token_logps(model, seqs, plens, pad)


@torch.inference_mode()
def dev_score(model_bf16, tok, dev, n, seed):
    prompts = [E.chat_ids(tok, E.mbpp_prompt(p)) for p in dev]
    outs = E.sample(model_bf16, tok, prompts, n, 160, 0.8, seed=seed)
    texts = [tok.decode(x, skip_special_tokens=True) for o in outs for x in o]
    g = E.grade_many(texts, [p for p in dev for _ in range(n)])
    p = sum(g) / len(g)
    return p, math.sqrt(max(p * (1 - p), 1e-4) / len(g))


_SR_GEN = torch.Generator().manual_seed(12345)


def to_bf16_stochastic(x):
    """Unbiased (stochastic) rounding of fp32 to bf16: weight changes smaller than bf16's step still reach the copy on
    average, instead of being rounded away."""
    xi = x.detach().contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, xi.shape, dtype=torch.int32, generator=_SR_GEN)
    return ((xi + r) & -65536).view(torch.float32).to(torch.bfloat16)


def sync(dst, src):
    with torch.no_grad():
        for a, b in zip(dst.parameters(), src.parameters()):
            a.copy_(to_bf16_stochastic(b) if a.dtype == torch.bfloat16 and b.dtype == torch.float32 else b.to(a.dtype))


def completion_ids(tok, code):
    return tok("```python\n" + code.strip() + "\n```", add_special_tokens=False)["input_ids"]


# ------------------------------------------------------------------------------------------------ updates
def grpo_update(a, model, ref, opt, items, pad):
    n_tok = sum(len(it["ids"]) - it["plen"] for it in items)
    model.train()
    opt.zero_grad(set_to_none=True)
    pg_sum = k3_sum = 0.0
    for s in range(0, len(items), a.micro):
        mb = items[s:s + a.micro]
        seqs, plens = [it["ids"] for it in mb], [it["plen"] for it in mb]
        with torch.inference_mode():
            rl = flat_lp(ref, seqs, plens, pad, a.autocast)
        rl = [x.clone() for x in rl]
        lp = flat_lp(model, seqs, plens, pad, a.autocast)
        loss = 0.0
        for x, r, it in zip(lp, rl, mb):
            d = r - x
            k3 = torch.exp(d) - d - 1
            pg = -it["adv"] * x
            loss = loss + (pg + a.beta * k3).sum()
            pg_sum += float(pg.detach().sum())
            k3_sum += float(k3.detach().sum())
        (loss / n_tok).backward()
    params = [p for p in model.parameters() if p.requires_grad]
    gn = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip))
    opt.step()
    model.eval()
    return dict(pg=pg_sum / n_tok, k3_ref=k3_sum / n_tok, grad_norm=gn, n_tok=n_tok)


class V8:
    """Update rule shared by V8 and V9: Adam-style direction (momentum + per-weight scale), split-half agreement logged,
    step size found by backtracking on the real model under KL caps."""

    def __init__(self, a, model):
        self.a = a
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        self.s_prev = a.s_init
        self.s_good = a.s_init

    def state(self):
        return dict(m=self.m, v=self.v, t=self.t, s_prev=self.s_prev, s_good=self.s_good)

    def load(self, st):
        self.v, self.t, self.s_prev = st["v"], st["t"], st["s_prev"]
        self.s_good = st.get("s_good", self.s_prev)
        if st.get("m") is not None:
            self.m = st["m"]

    def objective(self, model, pg, rs, old, n_pg, n_rs, pad, grad, half=None, clip=True):
        """PPO-clipped surrogate over the answers of `half` (None = all) + rescue log-likelihood. With grad=True it
        backpropagates -objective and returns nothing; with grad=False it returns (objective, k3 KL old->new, log-probs)."""
        a = self.a
        sel = [k for k, it in enumerate(pg) if half is None or it["half"] == half]
        rsel = [k for k, it in enumerate(rs) if half is None or it["half"] == half]
        obj = 0.0
        k3_sum = 0.0
        lps = [None] * len(pg)
        for s in range(0, len(sel), a.micro):
            ks = sel[s:s + a.micro]
            lp = flat_lp(model, [pg[k]["ids"] for k in ks], [pg[k]["plen"] for k in ks], pad, a.autocast and grad)
            part = 0.0
            for k, x in zip(ks, lp):
                ratio = torch.exp(x - old[k])
                adv = pg[k]["adv"]
                if clip:
                    term = torch.minimum(ratio * adv, ratio.clamp(1 - a.clip, 1 + a.clip) * adv)
                else:
                    term = ratio * adv
                part = part + term.sum()
                if not grad:
                    d = old[k] - x
                    k3_sum += float((torch.exp(d) - d - 1).sum())
                    lps[k] = x.float()
            part = part / n_pg
            if grad:
                if torch.is_tensor(part):
                    (-part).backward()
            else:
                obj += float(part)
        for s in range(0, len(rsel), a.micro):
            ks = rsel[s:s + a.micro]
            lp = flat_lp(model, [rs[k]["ids"] for k in ks], [rs[k]["plen"] for k in ks], pad, a.autocast and grad)
            part = a.rescue_weight * sum(x.sum() for x in lp) / max(n_rs, 1)
            if grad:
                (-part).backward()
            else:
                obj += float(part)
        if not grad:
            n_sel = sum(len(old[k]) for k in sel)
            return obj, k3_sum / max(n_sel, 1), lps

    def update(self, model, pg, rs, pres, pad, old=None, epochs=None, eps_old=None, eps_step=float("inf")):
        """V8: old=None (the current model wrote the answers), KL(old -> new) <= eps_old over all epochs.
        V9: old = the behaviour log-probs stored with the answers; KL(behaviour -> new) <= eps_old (staleness cap) and
        KL(current -> new) <= eps_step per update."""
        a = self.a
        epochs = a.epochs if epochs is None else epochs
        eps_old = a.eps_task if eps_old is None else eps_old
        rec = dict(epochs=[])
        n_pg = sum(len(it["ids"]) - it["plen"] for it in pg)
        n_rs = sum(len(it["ids"]) - it["plen"] for it in rs)
        if not pg and not rs:
            return rec
        if old is None:
            with torch.inference_mode():
                old = [x.float().clone() for x in flat_lp(model, [it["ids"] for it in pg], [it["plen"] for it in pg], pad, False)] if pg else []
        with torch.inference_mode():
            cur_obj, kl0, cur_lp = self.objective(model, pg, rs, old, n_pg, n_rs, pad, grad=False)
        rec["kl_old_start"] = kl0
        b1, b2 = a.beta1, a.beta2
        for ep in range(epochs):
            ga_all = None                                    # half A's gradient; half B's stays in .grad
            for h in (0, 1):
                model.zero_grad(set_to_none=True)
                model.train()
                self.objective(model, pg, rs, old, n_pg, n_rs, pad, grad=True, half=h)
                if h == 0:
                    ga_all = [(p.grad.detach().clone() if p.grad is not None else None) for p in self.params]
            self.t += 1
            c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
            dot = na = nb = 0.0
            for p, m, v, ga in zip(self.params, self.m, self.v, ga_all):
                gb = p.grad
                if ga is None and gb is None:
                    m.mul_(b1)
                    v.mul_(b2)
                    continue
                g = (ga if gb is None else (gb if ga is None else ga + gb))
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if ga is not None and gb is not None:
                    dot += float((ga * gb).sum())
                if ga is not None:
                    na += float((ga * ga).sum())
                if gb is not None:
                    nb += float((gb * gb).sum())
            ga_all = None
            model.zero_grad(set_to_none=True)
            model.eval()
            rho = dot / max(math.sqrt(na * nb), 1e-30)          # cosine between the two halves (logged)
            erec = dict(epoch=ep, rho=rho)
            if rho < a.rho_min:
                erec["result"] = "halves disagree: no step"
                rec["epochs"].append(erec)
                break
            s = min(a.s_max, 2 * self.s_prev)
            here = [p.detach().clone() for p in self.params]
            accepted = None
            tries = []
            for k in range(a.tries):
                with torch.no_grad():                           # .grad holds -d(objective): p = here - s * m_hat / (sqrt(v_hat) + 1e-8)
                    for p, h0, m, v in zip(self.params, here, self.m, self.v):
                        p.copy_(h0).addcdiv_(m, (v / c2).sqrt_().add_(1e-8), value=-s / c1)
                with torch.inference_mode():
                    new_obj, k3, new_lp = self.objective(model, pg, rs, old, n_pg, n_rs, pad, grad=False)
                ks = [(c - n_) for c, n_ in zip(cur_lp, new_lp) if c is not None]
                k3s = float(sum((torch.exp(d) - d - 1).sum() for d in ks) / max(sum(len(d) for d in ks), 1)) if ks else 0.0
                kl_ok = k3 <= eps_old and k3s <= eps_step
                klp = None
                if kl_ok and new_obj > cur_obj:                 # the chat check only for a step that passed the others
                    with torch.inference_mode():
                        klp = preserve_kl(model, pres)
                ok = kl_ok and new_obj > cur_obj and klp is not None and klp <= a.eps_pres
                tries.append(dict(s=s, obj=new_obj, kl_old=k3, kl_step=k3s, kl_pres=klp, ok=ok))
                if ok:
                    accepted = (s, new_obj, k3, klp, k3s, new_lp)
                    break
                if not kl_ok:                                   # KL grows ~ s^2: aim just under the tighter cap
                    r = min(eps_old / max(k3, 1e-12), eps_step / max(k3s, 1e-12))
                    s *= min(0.5, max(0.125, 0.9 * math.sqrt(r)))
                else:
                    s /= 2
            if accepted is None:
                with torch.no_grad():
                    for p, h0 in zip(self.params, here):
                        p.copy_(h0)
                self.s_prev = max(a.s_min, 0.25 * self.s_good if a.gentle_recovery else s)
                erec.update(result="no step passed the checks", tries=tries)
                rec["epochs"].append(erec)
                here = None
                break
            here = None
            r = min(eps_old / max(accepted[2], 1e-12), eps_step / max(accepted[4], 1e-12))
            self.s_prev = accepted[0] * min(1.0, max(0.25, 0.5 * math.sqrt(r)))   # next start = 2x this
            self.s_good = accepted[0]
            cur_obj, cur_lp = accepted[1], accepted[5]
            erec.update(result="step kept", s=accepted[0], obj=accepted[1], kl_task=accepted[2], kl_pres=accepted[3],
                        kl_step=accepted[4], tries=len(tries))
            rec["epochs"].append(erec)
        return rec


# ------------------------------------------------------------------------------------------------ V9 pool
def v9_refresh(a, tok, sampler_model, sampler, rng, bank, teacher):
    """Write answers in bulk with the current model, grade them once, keep the behaviour log-probs."""
    reload_teacher(teacher)
    ps = sampler.draw(a.pool_prompts, rng)
    prompts = [E.chat_ids(tok, E.mbpp_prompt(p)) for p in ps]
    outs, lps = E.sample(sampler_model, tok, prompts, a.pool_group, a.max_new, a.temperature, seed=rng.randrange(1 << 30),
                         rows_per_chunk=a.pool_chunk, return_logps=True)
    texts = [tok.decode(x, skip_special_tokens=True) for o in outs for x in o]
    g = E.grade_many(texts, [p for p in ps for _ in range(a.pool_group)])
    groups, rescues, n_ans, n_pass = [], [], 0, 0
    for i, (p, pr, o, lp) in enumerate(zip(ps, prompts, outs, lps)):
        r = g[i * a.pool_group:(i + 1) * a.pool_group]
        tid = p["task_id"]
        sampler.update(tid, sum(r), len(r))
        n_ans += len(r)
        n_pass += sum(r)
        for c, ok in zip(o, r):
            if ok:
                bank[tid] = (bank.get(tid, []) + [c])[-2:]
        R = torch.tensor(r, dtype=torch.float)
        if float(R.std()) > 1e-6:
            adv = ((R - R.mean()) / (R.std() + 1e-4)).tolist()
            groups.append(dict(uses=0, items=[dict(ids=pr + c, plen=len(pr), adv=av, old=l.float()) for c, av, l in zip(o, adv, lp)]))
        elif sum(r) == 0 and len(rescues) < a.pool_rescue:
            own = bank.get(tid, [])
            sol = own[-1] if own else (completion_ids(tok, teacher[str(tid)][0]) if teacher.get(str(tid)) else None)
            if sol is not None:
                rescues.append(dict(ids=pr + sol, plen=len(pr), own=bool(own)))
    return dict(groups=groups, rescues=rescues, r_ptr=0, fails=0, answers=n_ans, passes=n_pass, updates=0)


def v9_stale(a, pool):
    if pool is None or (not pool["groups"] and not pool["rescues"]):
        return True
    if pool["fails"] >= 2:
        return True
    return bool(pool["groups"]) and min(gr["uses"] for gr in pool["groups"]) >= a.reuse


def v9_minibatch(a, pool, rng):
    gs = sorted(pool["groups"], key=lambda gr: (gr["uses"], rng.random()))[:a.mb_groups]
    pg, old = [], []
    for j, gr in enumerate(gs):
        gr["uses"] += 1
        for it in gr["items"]:
            pg.append(dict(ids=it["ids"], plen=it["plen"], adv=it["adv"], half=j % 2))
            old.append(it["old"])
    rs = []
    R = pool["rescues"]
    for j in range(min(a.mb_rescue, len(R))):
        it = R[(pool["r_ptr"] + j) % len(R)]
        rs.append(dict(ids=it["ids"], plen=it["plen"], half=j % 2, own=it["own"]))
    pool["r_ptr"] += len(rs)
    return pg, rs, old


def v10_write(a, tok, model, sampler_model, sampler, rng, bank, teacher, pad, n_prompts):
    """V10 writer: stage 1 writes 2 answers per problem; stage 2 writes the other pool_group-2 only where the problem looks
    learnable (a pass in stage 1, or the sampler still expects mixed results). Mixed groups are then scored once by the
    training copy of the model, so the drift correction compares like with like."""
    k1 = a.stage1
    ps = sampler.draw(n_prompts, rng)
    prompts = [E.chat_ids(tok, E.mbpp_prompt(p)) for p in ps]
    o1 = E.sample(sampler_model, tok, prompts, k1, a.max_new, a.temperature, seed=rng.randrange(1 << 30), rows_per_chunk=a.pool_chunk)
    g1 = E.grade_many([tok.decode(x, skip_special_tokens=True) for o in o1 for x in o], [p for p in ps for _ in range(k1)])
    r1 = [g1[i * k1:(i + 1) * k1] for i in range(len(ps))]
    go = [i for i, p in enumerate(ps) if sum(r1[i]) > 0 or sampler.weight(p["task_id"]) >= a.stage2_min_weight]
    k2 = a.pool_group - k1
    o2 = E.sample(sampler_model, tok, [prompts[i] for i in go], k2, a.max_new, a.temperature, seed=rng.randrange(1 << 30),
                  rows_per_chunk=a.pool_chunk) if go and k2 > 0 else []
    g2 = E.grade_many([tok.decode(x, skip_special_tokens=True) for o in o2 for x in o], [ps[i] for i in go for _ in range(k2)]) if o2 else []
    comps = {i: list(o1[i]) for i in range(len(ps))}
    rews = {i: list(r1[i]) for i in range(len(ps))}
    for j, i in enumerate(go):
        comps[i] += o2[j]
        rews[i] += g2[j * k2:(j + 1) * k2]
    groups, rescues, n_ans, n_pass = [], [], 0, 0
    for i, p in enumerate(ps):
        tid, r, o, pr = p["task_id"], rews[i], comps[i], prompts[i]
        sampler.update(tid, sum(r), len(r))
        n_ans += len(r)
        n_pass += sum(r)
        for c, ok in zip(o, r):
            if ok:
                bank[tid] = (bank.get(tid, []) + [c])[-2:]
        R = torch.tensor(r, dtype=torch.float)
        if len(r) > k1 and float(R.std()) > 1e-6:
            adv = ((R - R.mean()) / (R.std() + 1e-4)).tolist()
            groups.append(dict(uses=0, items=[dict(ids=pr + c, plen=len(pr), adv=av) for c, av in zip(o, adv)]))
        elif sum(r) == 0:
            own = bank.get(tid, [])
            sol = own[-1] if own else (completion_ids(tok, teacher[str(tid)][0]) if teacher.get(str(tid)) else None)
            if sol is not None:
                rescues.append(dict(ids=pr + sol, plen=len(pr), own=bool(own), tid=tid))
    items = [it for gr in groups for it in gr["items"]]
    if items:
        with torch.inference_mode():
            old = flat_lp(model, [it["ids"] for it in items], [it["plen"] for it in items], pad, False)
        for it, l in zip(items, old):
            it["old"] = l.float().clone()
    return groups, rescues, n_ans, n_pass, len(go), len(ps)


NO_TEACHER = False


def reload_teacher(teacher):
    """Pick up teacher.json once it exists (bank.py may still be writing it when training starts)."""
    tp = os.path.join(DATA, "teacher.json")
    if not teacher and os.path.exists(tp) and not NO_TEACHER:
        teacher.update(json.load(open(tp)))
        E.log(f"teacher bank loaded: {sum(bool(v) for v in teacher.values())} problems with a test-passing solution")


def v10_maintain(a, pool, tok, model, sampler_model, sampler, rng, bank, teacher, pad):
    """Rolling store: when the store is small, or a quarter of it has been used --reuse times, or two updates in a row
    failed their checks, write a fresh chunk and drop the most-used groups beyond --pool-groups."""
    reload_teacher(teacher)
    if pool is None:
        pool = dict(groups=[], rescues=[], r_ptr=0, fails=0, answers=0, passes=0, updates=0)
    used = sum(gr["uses"] >= a.reuse for gr in pool["groups"])
    need = len(pool["groups"]) < a.mb_groups or used >= max(1, len(pool["groups"]) // 4) or pool["fails"] >= 2
    if not need:
        return pool, None
    sync(sampler_model, model)
    groups, rescues, n_ans, n_pass, n_go, n_ps = v10_write(a, tok, model, sampler_model, sampler, rng, bank, teacher, pad,
                                                          a.chunk_prompts)
    keep = sorted(pool["groups"], key=lambda gr: gr["uses"])
    pool["groups"] = (groups + keep)[:a.pool_groups]
    seen = {r["tid"] for r in rescues}
    pool["rescues"] = (rescues + [r for r in pool["rescues"] if r["tid"] not in seen])[:a.pool_rescue]
    pool["fails"] = 0
    pool["answers"] += n_ans
    pool["passes"] += n_pass
    return pool, dict(written=n_ans, passed=n_pass, new_groups=len(groups), stage2=f"{n_go}/{n_ps}", rescues=len(rescues))


# ------------------------------------------------------------------------------------------------ main loop
def run(a):
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    tok, model = E.load()
    _, sampler_model = E.load(dtype=torch.bfloat16)
    pad = tok.pad_token_id if tok.pad_token_id is not None else E.sc.eos_ids(tok)[0]
    train = E.problems("train")
    dev = E.problems("validation")
    if a.method == "v14" or a.pick_split:  # choices are made on held-out training problems; validation/test stay untouched
        sh = list(train)
        random.Random(1234).shuffle(sh)
        dev, train = sh[-a.pick_size:], sh[:-a.pick_size]
        E.log(f"V14: {len(train)} training problems, {len(dev)} held-out problems for choosing checkpoints and jumps")
    by_id = {p["task_id"]: p for p in train}
    screen = json.load(open(os.path.join(DATA, "screen.json")))
    sampler = Sampler(train, screen, a.group)
    E.log(f"sampler prior Beta({sampler.prior[0]:.3f}, {sampler.prior[1]:.2f}); share of draws on never-solved problems "
          f"{sampler.share_never_solved():.0%}")
    tp = os.path.join(DATA, "teacher.json")
    teacher = json.load(open(tp)) if a.method != "grpo" and os.path.exists(tp) and not a.no_teacher else {}
    if a.method != "grpo" and not teacher:
        E.log("WARNING: no teacher.json; rescues use the model's own past successes only")
    pres_path = os.path.join(a.out, "pres.pt")
    ref = opt = v8 = None
    if a.method == "grpo":
        _, ref = E.load()
        ref.requires_grad_(False)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    else:
        v8 = V8(a, model)
        if os.path.exists(pres_path):
            pres = torch.load(pres_path, weights_only=False)
        else:
            pres = build_preserve(tok, model)
            torch.save(pres, pres_path)
    st = E.load_ckpt(a.out)
    rnd, compute_s, bank = 0, 0.0, {}
    best_dev = None
    rng = random.Random(a.seed)
    if st is not None:
        model.load_state_dict(st["model"])
        rnd, compute_s, bank, best_dev = st["round"], st["compute_s"], st["bank"], st["best_dev"]
        sampler.load(st["sampler"])
        rng.setstate(st["rng"])
        torch.set_rng_state(st["torch_rng"])
        if opt is not None:
            opt.load_state_dict(st["opt"])           # load_state_dict casts the averages back to the parameters' dtype
        if v8 is not None:
            st["v8"]["m"] = [x.float() for x in st["v8"]["m"]]
            st["v8"]["v"] = [x.float() for x in st["v8"]["v"]]
            v8.load(st["v8"])
        E.log(f"resumed at round {rnd}, compute {compute_s / 60:.1f} min")
        logp = os.path.join(a.out, "train_log.jsonl")
        if os.path.exists(logp):
            keep = [l for l in open(logp) if json.loads(l).get("round", 0) <= rnd]
            open(logp, "w").writelines(keep)
    else:
        json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a")
    last_save = time.time()

    def half(x):                                   # optimizer averages are stored in bf16 to save disk
        if torch.is_tensor(x) and x.is_floating_point() and x.numel() > 1:
            return x.to(torch.bfloat16)
        if isinstance(x, dict):
            return {k: half(v) for k, v in x.items()}
        if isinstance(x, list):
            return [half(v) for v in x]
        return x

    def checkpoint():
        state = dict(round=rnd, compute_s=compute_s, model=model.state_dict(), bank=bank, best_dev=best_dev,
                     sampler=sampler.state(), rng=rng.getstate(), torch_rng=torch.get_rng_state(),
                     opt=half(opt.state_dict()) if opt is not None else None, v8=half(v8.state()) if v8 is not None else None)
        E.save_ckpt(a.out, state)

    def do_dev():
        nonlocal best_dev
        sync(sampler_model, model)
        p, se = dev_score(sampler_model, tok, dev, a.dev_n, seed=(999 + rnd) if (a.method == "v14" or a.pick_split) else 999)
        rec = dict(dev=p, dev_se=se)
        if best_dev is None or p > best_dev["p"]:
            best_dev = dict(p=p, se=se, round=rnd)
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt.tmp"))
            os.replace(os.path.join(a.out, "best.pt.tmp"), os.path.join(a.out, "best.pt"))
            rec["new_best"] = True
        elif a.method != "grpo" and p < best_dev["p"] - 2 * math.sqrt(se ** 2 + best_dev["se"] ** 2):
            model.load_state_dict(torch.load(os.path.join(a.out, "best.pt"), weights_only=False))
            v8.s_prev = max(a.s_min, v8.s_prev / 4)
            rec["rolled_back_to_round"] = best_dev["round"]
        return rec

    if best_dev is None:
        t = time.time()
        rec = do_dev()
        compute_s += time.time() - t
        rec.update(round=0, compute_min=compute_s / 60)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        E.log(f"round 0 dev {rec['dev']:.3f} ± {rec['dev_se']:.3f}")

    pool = None
    while compute_s < a.budget_min * 60:
        t0 = time.time()
        rnd += 1
        if a.method in ("v9", "v10", "v14"):
            wrote = None
            if a.method == "v9":
                if v9_stale(a, pool):
                    sync(sampler_model, model)
                    pool = v9_refresh(a, tok, sampler_model, sampler, rng, bank, teacher)
                    wrote = dict(written=pool["answers"], passed=pool["passes"], new_groups=len(pool["groups"]),
                                 rescues=len(pool["rescues"]))
            else:
                pool, wrote = v10_maintain(a, pool, tok, model, sampler_model, sampler, rng, bank, teacher, pad)
            t_roll = time.time() - t0
            t1 = time.time()
            pg, rs, old = v9_minibatch(a, pool, rng)
            upd = v8.update(model, pg, rs, pres, pad, old=old, epochs=1, eps_old=a.eps_pool, eps_step=a.eps_task)
            ep = upd.get("epochs", [])
            pool["fails"] = pool["fails"] + 1 if (ep and ep[-1]["result"] != "step kept") else 0
            pool["updates"] += 1
            t_upd = time.time() - t1
            rec = dict(round=rnd, wrote=wrote, pool_groups=len(pool["groups"]), pool_rescues=len(pool["rescues"]),
                       mb_answers=len(pg), mb_rescues=len(rs), t_roll=round(t_roll, 1), t_update=round(t_upd, 1), update=upd)
            if rnd % a.dev_every == 0:
                rec.update(do_dev())
            compute_s += time.time() - t0
            rec["compute_min"] = compute_s / 60
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            msg = f"update {rnd} | " + (f"wrote {wrote['written']} answers ({wrote['passed']} pass, {wrote['new_groups']} useful groups"
                                        + (f", stage 2 on {wrote['stage2']}" if 'stage2' in wrote else "") + f") in {t_roll:.0f}s | "
                                        if wrote else "") + \
                  f"store {len(pool['groups'])} groups + {len(pool['rescues'])} known-good | update {t_upd:.0f}s | " + \
                  "; ".join(f"rho {e['rho']:.3f} {e['result']}" + (f" s {e['s']:.1e} kl_step {e['kl_step']:.4f} kl_stale {e['kl_task']:.4f} pres {e['kl_pres']:.4f}" if "s" in e else "") for e in ep) + \
                  f" | compute {compute_s / 60:.1f}/{a.budget_min} min | peak mem {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB"
            if "dev" in rec:
                msg += f" | DEV {rec['dev']:.3f} ± {rec['dev_se']:.3f}" + (" best" if rec.get("new_best") else "") + \
                       (f" ROLLBACK to {rec['rolled_back_to_round']}" if "rolled_back_to_round" in rec else "")
                if "rolled_back_to_round" in rec:
                    pool = None
            E.log(msg)
            if time.time() - last_save > a.save_every_s:
                checkpoint()
                last_save = time.time()
            continue
        sync(sampler_model, model)
        batch = sampler.draw(a.prompts, rng)
        groups = []

        def roll(ps):
            prompts = [E.chat_ids(tok, E.mbpp_prompt(p)) for p in ps]
            outs = E.sample(sampler_model, tok, prompts, a.group, a.max_new, a.temperature, seed=rng.randrange(1 << 30))
            texts = [tok.decode(x, skip_special_tokens=True) for o in outs for x in o]
            g = E.grade_many(texts, [p for p in ps for _ in range(a.group)])
            for i, (p, pr, o) in enumerate(zip(ps, prompts, outs)):
                groups.append(dict(p=p, prompt=pr, comps=o, r=g[i * a.group:(i + 1) * a.group]))

        roll(batch)
        if a.method == "v8":
            while sum(0 < sum(gr["r"]) < a.group for gr in groups) < a.min_inf and len(groups) < a.max_prompts:
                roll(sampler.draw(a.extra_prompts, rng, exclude={gr["p"]["task_id"] for gr in groups}))
        t_roll = time.time() - t0
        pg, rs = [], []
        n_mixed = n_fail = n_rescued = 0          # halves alternate over mixed groups, and separately over rescues
        for gi, gr in enumerate(groups):
            tid = gr["p"]["task_id"]
            sampler.update(tid, sum(gr["r"]), len(gr["r"]))
            for c, r in zip(gr["comps"], gr["r"]):
                if r:
                    bank[tid] = (bank.get(tid, []) + [c])[-2:]
            R = torch.tensor(gr["r"], dtype=torch.float)
            if float(R.std()) > 1e-6:
                n_mixed += 1
                adv = ((R - R.mean()) / (R.std() + 1e-4)).tolist()
                for c, av in zip(gr["comps"], adv):
                    pg.append(dict(ids=gr["prompt"] + c, plen=len(gr["prompt"]), adv=av, half=n_mixed % 2))
            elif sum(gr["r"]) == 0:
                n_fail += 1
                if a.method == "v8" and n_rescued < a.max_rescue:
                    own = [c for c in bank.get(tid, [])]
                    sol = own[-1] if own else (completion_ids(tok, teacher[str(tid)][0]) if teacher.get(str(tid)) else None)
                    if sol is not None:
                        rs.append(dict(ids=gr["prompt"] + sol, plen=len(gr["prompt"]), half=n_rescued % 2, own=bool(own)))
                        n_rescued += 1
        t1 = time.time()
        if a.method == "grpo":
            upd = grpo_update(a, model, ref, opt, pg, pad) if pg else dict(skipped=True)
        else:
            upd = v8.update(model, pg, rs, pres, pad)
        t_upd = time.time() - t1
        rewards = [r for gr in groups for r in gr["r"]]
        rec = dict(round=rnd, answers=len(rewards), pass_rate=sum(rewards) / len(rewards), mixed=n_mixed, all_fail=n_fail,
                   rescued=n_rescued, rescued_own=sum(x.get("own", False) for x in rs), t_roll=round(t_roll, 1),
                   t_update=round(t_upd, 1), update=upd,
                   mean_len=sum(len(c) for gr in groups for c in gr["comps"]) / len(rewards))
        if rnd % a.dev_every == 0:
            rec.update(do_dev())
        compute_s += time.time() - t0
        rec["compute_min"] = compute_s / 60
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        msg = (f"round {rnd} | {len(rewards)} answers, pass {rec['pass_rate']:.3f}, mixed {n_mixed}, all-fail {n_fail}, "
               f"rescued {n_rescued} | roll {t_roll:.0f}s update {t_upd:.0f}s | compute {compute_s / 60:.1f}/{a.budget_min} min | "
               f"peak mem {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
        if a.method == "v8":
            msg += " | " + "; ".join(f"e{e['epoch']} rho {e['rho']:.3f} {e['result']}" + (f" s {e['s']:.1e} kl {e['kl_task']:.4f} pres {e['kl_pres']:.4f}" if "s" in e else "") for e in upd.get("epochs", []))
        else:
            msg += f" | k3_ref {upd.get('k3_ref', 0):.5f}"
        if "dev" in rec:
            msg += f" | DEV {rec['dev']:.3f} ± {rec['dev_se']:.3f}" + (" best" if rec.get("new_best") else "") + \
                   (f" ROLLBACK to round {rec['rolled_back_to_round']}" if "rolled_back_to_round" in rec else "")
        E.log(msg)
        if time.time() - last_save > a.save_every_s:
            checkpoint()
            last_save = time.time()
    checkpoint()
    # final model = best on validation (the last weights if nothing beat the start)
    final = os.path.join(a.out, "final")
    if best_dev is not None and best_dev["round"] > 0 and os.path.exists(os.path.join(a.out, "best.pt")):
        model.load_state_dict(torch.load(os.path.join(a.out, "best.pt"), weights_only=False))
        chosen = f"best on validation (round {best_dev['round']}, dev {best_dev['p']:.3f})"
    else:
        chosen = "nothing beat the starting model on validation; final = starting weights"
        model = E.load()[1]
    jump = None
    if a.method == "v14" and best_dev is not None and best_dev["round"] > 0:
        # one jump along the whole run's change, W_base + k (W_best - W_base), within the chat limit, chosen on the
        # held-out training problems with one fresh seed shared by all candidates
        base_sd = E.load()[1].state_dict()
        best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        tried = []
        for k in [1.0] + a.end_jumps:
            with torch.no_grad():
                model.load_state_dict({n: base_sd[n] + k * (best_sd[n] - base_sd[n]) for n in best_sd})
            klp = preserve_kl(model, pres)
            sync(sampler_model, model)
            pk, _ = dev_score(sampler_model, tok, dev, a.jump_n, seed=4242)
            tried.append(dict(k=k, held_out=pk, chat_change=klp))
            E.log(f"end jump {k}x the run's change: held-out {pk:.3f}, chat change {klp:.4f}")
        ok = [t for t in tried if t["chat_change"] <= a.jump_chat_cap] or tried[:1]
        pick = max(ok, key=lambda t: t["held_out"])
        with torch.no_grad():
            model.load_state_dict({n: base_sd[n] + pick["k"] * (best_sd[n] - base_sd[n]) for n in best_sd})
        jump = dict(tried=tried, chosen=pick)
        chosen += f"; then jumped {pick['k']}x along the run's change (held-out {pick['held_out']:.3f})"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    json.dump(dict(chosen=chosen, rounds=rnd, compute_min=compute_s / 60, jump=jump), open(os.path.join(a.out, "final.json"), "w"))
    open(os.path.join(a.out, "DONE"), "w").write("done\n")
    E.log(f"finished: {rnd} rounds, {compute_s / 60:.1f} compute min; final model = {chosen}")


# ------------------------------------------------------------------------------------------------ selfcheck
def selfcheck(a):
    """(1) V8's objective at epoch 0 has the same gradient as GRPO's policy term (beta = 0);
    (2) the two half-gradients add up to the full one; (3) preserve KL is 0 at the base and > 0 after a perturbation;
    (4) bf16 noise floor of the log-probs used by the KL checks."""
    torch.set_num_threads(a.threads)
    tok, model = E.load()
    pad = tok.pad_token_id
    probs = E.problems("train")[:4]
    prompts = [E.chat_ids(tok, E.mbpp_prompt(p)) for p in probs]
    outs = E.sample(model, tok, prompts, 3, 48, 1.0, seed=1)
    pg = []
    for i, (pr, o) in enumerate(zip(prompts, outs)):
        for j, c in enumerate(o):
            pg.append(dict(ids=pr + c, plen=len(pr), adv=[1.0, -0.5, -0.5][j] * (1 if i % 2 else -1), half=i % 2))
    a.autocast = False
    v8 = V8(a, model)
    n_pg = sum(len(it["ids"]) - it["plen"] for it in pg)
    with torch.inference_mode():
        old = [x.clone() for x in E.token_logps(model, [it["ids"] for it in pg], [it["plen"] for it in pg], pad)]
    model.zero_grad(set_to_none=True)
    v8.objective(model, pg, [], old, n_pg, 0, pad, grad=True)
    g_full = [p.grad.clone() for p in v8.params]
    halves = []
    for h in (0, 1):
        model.zero_grad(set_to_none=True)
        v8.objective(model, pg, [], old, n_pg, 0, pad, grad=True, half=h)
        halves.append([p.grad.clone() for p in v8.params])
    model.zero_grad(set_to_none=True)
    _, ref = E.load()
    a.beta, a.grad_clip = 0.0, 1e12
    opt = torch.optim.SGD(v8.params, lr=0.0)
    grpo_update(a, model, ref, opt, pg, pad)
    g_grpo = [p.grad.clone() for p in v8.params]
    rel = lambda x, y: float(torch.sqrt(sum(((u - w) ** 2).sum() for u, w in zip(x, y))) / torch.sqrt(sum((w ** 2).sum() for w in y)))
    r1 = rel(g_full, g_grpo)
    r2 = rel([x + y for x, y in zip(*halves)], g_full)
    pres = build_preserve(tok, model, n=4, reply=24)
    k0 = preserve_kl(model, pres)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(1 + 0.01 * torch.randn_like(p))
    k1 = preserve_kl(model, pres)
    with torch.inference_mode():
        lp32 = E.token_logps(model, [it["ids"] for it in pg], [it["plen"] for it in pg], pad)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lp16 = E.token_logps(model, [it["ids"] for it in pg], [it["plen"] for it in pg], pad)
    d = torch.cat([x - y.float() for x, y in zip(lp32, lp16)])
    floor = float((torch.exp(d) - d - 1).mean())
    E.log(f"(1) V8 vs GRPO gradient rel err {r1:.2e} -> {'PASS' if r1 < 1e-5 else 'FAIL'}")
    E.log(f"(2) half-gradients sum vs full rel err {r2:.2e} -> {'PASS' if r2 < 1e-5 else 'FAIL'}")
    E.log(f"(3) preserve KL at base {k0:.2e}, after 1% noise {k1:.3f} -> {'PASS' if k0 < 1e-6 and k1 > 1e-3 else 'FAIL'}")
    E.log(f"(4) bf16 vs fp32 log-prob gap: mean |d| {float(d.abs().mean()):.4f}, k3 noise floor {floor:.2e} "
          f"(vs --eps-task {a.eps_task})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["granderl", "grpo", "v8", "v9", "v10", "v14"],
                    help="granderl = GrandeRL-1.0 (the method developed as V14); grpo = plain RL baseline")
    ap.add_argument("--out")
    ap.add_argument("--budget-min", type=float, default=240)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--autocast", type=int, default=1, help="bf16 autocast for training forwards/backwards")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-every", type=int, default=10)
    ap.add_argument("--dev-n", type=int, default=4)
    ap.add_argument("--save-every-s", type=float, default=300)
    # grpo
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # v8
    ap.add_argument("--min-inf", type=int, default=6)
    ap.add_argument("--max-prompts", type=int, default=24)
    ap.add_argument("--extra-prompts", type=int, default=6)
    ap.add_argument("--max-rescue", type=int, default=4)
    ap.add_argument("--rescue-weight", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    # v9 / v10 (store of graded answers, many updates per writing pass)
    ap.add_argument("--pool-prompts", type=int, default=96, help="v9: problems per bulk rewrite")
    ap.add_argument("--pool-group", type=int, default=8)
    ap.add_argument("--pool-chunk", type=int, default=128)
    ap.add_argument("--pool-rescue", type=int, default=16)
    ap.add_argument("--reuse", type=int, default=3)
    ap.add_argument("--mb-groups", type=int, default=6)
    ap.add_argument("--mb-rescue", type=int, default=2)
    ap.add_argument("--eps-pool", type=float, default=0.02, help="cap on KL(writer -> current) over stored answers")
    ap.add_argument("--chunk-prompts", type=int, default=32, help="v10: problems per rolling write")
    ap.add_argument("--pool-groups", type=int, default=30, help="v10: groups kept in the store")
    ap.add_argument("--stage1", type=int, default=2)
    ap.add_argument("--stage2-min-weight", type=float, default=0.3)
    ap.add_argument("--rho-min", type=float, default=-1.0, help="skip a step when the halves' cosine is below this (-1 = never)")
    ap.add_argument("--eps-task", type=float, default=0.003)
    ap.add_argument("--eps-pres", type=float, default=0.1)
    ap.add_argument("--s-init", type=float, default=5e-6)
    ap.add_argument("--s-min", type=float, default=1e-7)
    ap.add_argument("--s-max", type=float, default=1e-4)
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--pick-size", type=int, default=74, help="v14: training problems held out for choices")
    ap.add_argument("--pick-split", type=int, default=0, help="any method: use V14's split (300 train / 74 held out, fresh seeds)")
    ap.add_argument("--no-teacher", type=int, default=0, help="ignore teacher.json (known-good answers only from own successes)")
    ap.add_argument("--gentle-recovery", type=int, default=0, help="after a rejected step restart at 1/4 of the last good one")
    ap.add_argument("--end-jumps", type=lambda s: [float(x) for x in s.split(",")], default=[1.5, 2.0, 3.0, 5.0])
    ap.add_argument("--jump-chat-cap", type=float, default=0.05)
    ap.add_argument("--jump-n", type=int, default=8, help="v14: answers per held-out problem when choosing the end jump")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.method == "granderl":            # GrandeRL-1.0 is the method developed as V14
        a.method = "v14"
    global NO_TEACHER
    NO_TEACHER = bool(a.no_teacher)
    if a.method == "v14":                 # V14 defaults: V10's update with larger steps, gentler recovery
        for name, val in (("eps-task", 0.015), ("eps-pool", 0.08), ("s-max", 3e-4), ("gentle-recovery", 1)):
            if f"--{name}" not in sys.argv:
                setattr(a, name.replace("-", "_"), val)
    a.autocast = bool(a.autocast)
    torch.manual_seed(a.seed)
    if a.selfcheck:
        selfcheck(a)
        return
    if not (a.method and a.out):
        ap.error("--method and --out are required")
    run(a)


if __name__ == "__main__":
    main()
