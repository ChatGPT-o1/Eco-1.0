"""agent.py - a ~1K-parameter neural network learns, by reinforcement, how to edit SmolLM-135M so it generates faster.

EPISODES. Every episode starts from the original model. Each step the network looks at every edit that is still
possible anywhere in the architecture and picks one:
    attn i      remove the attention sublayer of layer i
    mlp i       remove the MLP sublayer of layer i
    p10 / p30   remove another 10% / 30% of layer i's MLP neurons (the least-used ones)
    kv i g      remove key/value head group g of layer i (with its 3 query heads)
    half        run the weights in 16-bit floats (GPU only)
Weights are only kept, removed or stored in fewer bits. Nothing is retrained.
An edit that would push search perplexity past the budget is refused (a strike, small penalty); three strikes end the
episode. So every episode ends on a model that is inside the budget, and its return is that model's speed-up.

REWARD (fixed; the network cannot change it): 10 x the speed gained by each accepted edit, -0.2 per refused edit.
Speed during an episode comes from a latency model fitted to real decoder timings on this machine at start-up.
Perplexity is exact (smol.py) and judged pessimistically: mean increase + one standard error must stay within 10%.
The search text is split into subsets and every episode is scored on the next one, so the network cannot profit
from the noise of one particular piece of text.

LEARNING. PPO-style policy gradient on the network's own episodes, with a small linear baseline.

SAFEGUARDS (scoring side, outside the agent): an architecture becomes the saved best only if it is within budget on
the WHOLE search text, VALIDATION perplexity is also within 10%, AND its speed, timed for real with the decoder, beats
the best so far. Test text is used once, for the final report.
"""
import os, json, math, time, copy, threading, statistics
import torch, torch.nn as nn
import judge as J, smol as S, fastgen as G

LOOP = float(os.environ.get('LOOP_SECONDS', 300))
OUT = os.environ.get('OUT_DIR', f'{J.ROOT}/runs_v3'); os.makedirs(OUT, exist_ok=True)
SEED = int(os.environ.get('SEED', 0))
TOL = 1.10; BUDGET = math.log(TOL)
TMAX, STRIKES = 60, 3
ROUND_EPISODES = int(os.environ.get('ROUND_EPISODES', 4)); SUBSETS = int(os.environ.get('SUBSETS', 4))
CAL_RUNS = int(os.environ.get('CAL_RUNS', 5)); CAL_TOKENS = int(os.environ.get('CAL_TOKENS', 64))   # decoder timing: best of N runs of T tokens
LR, EPOCHS, CLIP, ENT = 0.01, 4, 0.2, 0.01
TIMING_TOL = float(os.environ.get('TIMING_TOL', 0.01))
# fast scoring: predict from the single-edit probe costs; measure on a small slice; full measurement only near the line
FAST = int(os.environ.get('FAST', 1)); PRED_LO, PRED_HI = 0.5, 2.0; SCREEN_N = int(os.environ.get('SCREEN_N', 16)); GATE_EVERY = int(os.environ.get('GATE_EVERY', 5)); SCREEN_Z = 2.0      # real speed must confirm the latency model within this
FEAT = 18
T0 = time.time()


def say(**k):
    k['t'] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)


# ----------------------------------------------------------------------------------------------- the 1K network
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEAT, 32), nn.Tanh(), nn.Linear(32, 12), nn.Tanh(), nn.Linear(12, 1))
        with torch.no_grad(): self.net[-1].weight.mul_(0.1); self.net[-1].bias.zero_()

    def forward(self, f): return self.net(f).squeeze(-1)


# ----------------------------------------------------------------------------------------------- latency model
class Latency:
    """ms per token = fixed + per kept attention sublayer (A0 + A1 x share of head groups kept)
                            + per kept MLP sublayer (M0 + M1 x width).   Five numbers per precision."""

    def __init__(self, nkv): self.coef = {}; self.nkv = nkv; self.ms0 = None

    def ms(self, a):
        c, A0, A1, M0, M1 = self.coef[a.half]; t = c
        for i in range(len(a.attn)):
            if a.attn[i]: t += A0 + A1 * bin(a.kv[i]).count('1') / self.nkv
            if a.w[i] >= 0: t += M0 + M1 * S.WIDTHS[a.w[i]]
        return t

    def speedup(self, a): return self.ms0 / self.ms(a)


def calibrate(w, halves):
    """Fit the latency model from real decoder timings. Every architecture that fits the perplexity budget is close
    to the full model, so the model is fitted there: time the full model, then the full model with one kind of part
    taken out of a third of the layers (or thinned in all of them). A mixed architecture that was not used for the
    fit checks the prediction."""
    net = w.n32; nl, nkv = net.NL, net.nkv; lat = Latency(nkv); info = {}; k3 = [i for i in range(nl) if i % 3 == 1]
    for half in halves:
        def mk(fn):
            a = net.full(); a.half = half
            for i in range(nl): fn(a, i)
            return a
        def mixed(a, i):
            if i % 6 == 0: a.attn[i] = False
            elif i % 6 == 3: a.kv[i] = 0b011
            if i % 10 == 5: a.w[i] = -1
            elif i % 5 == 1: a.w[i] = 3
            elif i % 5 == 2: a.w[i] = 5
        archs = dict(full=mk(lambda a, i: None), attn_third=mk(lambda a, i: a.attn.__setitem__(i, i % 3 != 1)),
                     mlp_third=mk(lambda a, i: a.w.__setitem__(i, -1 if i % 3 == 1 else 0)),
                     mlp50=mk(lambda a, i: a.w.__setitem__(i, 5)), kv1=mk(lambda a, i: a.kv.__setitem__(i, 1)), mixed=mk(mixed))
        t = {k: 1000.0 / w.tok_s(a, CAL_RUNS, CAL_TOKENS) for k, a in archs.items()}
        A = (t['full'] - t['attn_third']) / len(k3); M = (t['full'] - t['mlp_third']) / len(k3)
        M1 = (t['full'] - t['mlp50']) / (0.5 * nl); A1 = (t['full'] - t['kv1']) / (nl * (1 - 1 / nkv))
        lat.coef[half] = (t['full'] - nl * (A + M), A - A1, A1, M - M1, M1)
        pred = lat.ms(archs['mixed'])
        info['fp16' if half else 'fp32'] = dict(ms={k: round(v, 3) for k, v in t.items()},
            attn_sublayer_ms=round(A, 4), kv_group_ms=round(A1 / nkv, 4), mlp_sublayer_ms=round(M, 4), mlp_width_part_ms=round(M1, 4),
            check_pred_ms=round(pred, 3), check_real_ms=round(t['mixed'], 3), check_error_pct=round(100 * (pred / t['mixed'] - 1), 2))
    lat.ms0 = lat.ms(net.full())
    return lat, info


# ----------------------------------------------------------------------------------------------- shared knowledge
class Shared:
    """What every worker reads: the latency model, usage statistics and single-edit probe costs of the original
    model, and a memo of architectures already scored."""
    def __init__(self): self.memo = {}; self.evals = 0; self.hits = 0; self.predicted = 0; self.screened = 0; self.overshoot = 0; self.allow_half = False


def edit_cost(arch, e, sh):
    """Predicted increase of search NLL per token from one edit: its cost when probed alone on the original model."""
    k = e[0]
    if k == 'half': return 0.0
    i = e[1]; wi = arch.w[i]; nk = bin(arch.kv[i]).count('1')
    if k == 'attn': return sh.p_attn[i] * (nk / sh.nkv) ** 0.5
    if k == 'mlp': return sh.p_mlp[i] - sh.p_w[i][wi]
    if k == 'kv': return sh.p_kv[i][e[2]]
    nw = wi + (1 if k == 'p10' else 3); return sh.p_w[i][nw] - sh.p_w[i][wi]


def candidates(arch, blocked, allow_half, nkv):
    c = []
    for i in range(len(arch.attn)):
        if arch.attn[i]:
            c.append(('attn', i))
            if bin(arch.kv[i]).count('1') > 1:
                c.extend(('kv', i, g) for g in range(nkv) if arch.kv[i] >> g & 1)
        if arch.w[i] >= 0:
            c.append(('mlp', i))
            if arch.w[i] + 1 < len(S.WIDTHS): c.append(('p10', i))
            if arch.w[i] + 3 < len(S.WIDTHS): c.append(('p30', i))
    if allow_half and not arch.half: c.append(('half',))
    return [e for e in c if e not in blocked] if blocked else c


def features(arch, cands, used, s_cur, strikes, sh):
    """One row of FEAT numbers per possible edit: what it is, where it is, the state of that layer, what the latency
    model says it saves, what it cost when probed alone on the original model, how much that part writes into the
    residual stream, and the episode's state (budget used, speed so far, strikes)."""
    lat = sh.lat; nkv = lat.nkv; nl = len(arch.attn); ms0 = lat.ms0; ms_cur = lat.ms(arch)
    c, A0, A1, M0, M1 = lat.coef[arch.half]
    g_state = [min(1.5, max(0.0, used / BUDGET)), s_cur - 1, strikes / STRIKES, 1.0 if arch.half else 0.0]
    rows, gains = [], []
    for e in cands:
        k = e[0]; f = [0.0] * FEAT
        if k == 'half':
            f[4] = 1.0; ms_new = lat.ms(arch.apply(e)); cost = 0.0; wr = 0.0; rem = 0.0
        else:
            i = e[1]; nk = bin(arch.kv[i]).count('1'); wi = arch.w[i]; wcur = S.WIDTHS[wi] if wi >= 0 else 0.0
            if k == 'attn':
                f[0] = 1.0; d = A0 + A1 * nk / nkv; cost = sh.p_attn[i]; wr = sh.aw[i] * nk / nkv; rem = nk / nkv
            elif k == 'mlp':
                f[1] = 1.0; d = M0 + M1 * wcur; cost = sh.p_mlp[i] - sh.p_w[i][wi]; wr = sh.mw[i] * (1 - sh.mass[i][wi]); rem = wcur
            elif k == 'kv':
                f[3] = 1.0; d = A1 / nkv; cost = sh.p_kv[i][e[2]]; wr = sh.kw[i][e[2]]; rem = 1 / nkv
            else:
                f[2] = 1.0; nw = wi + (1 if k == 'p10' else 3); d = M1 * (wcur - S.WIDTHS[nw])
                cost = sh.p_w[i][nw] - sh.p_w[i][wi]; wr = sh.mw[i] * (sh.mass[i][nw] - sh.mass[i][wi]); rem = wcur - S.WIDTHS[nw]
            ms_new = ms_cur - d
            f[5] = i / (nl - 1); f[10] = wcur; f[11] = nk / nkv if arch.attn[i] else 0.0
            nb = [(not arch.attn[j]) if k in ('attn', 'kv') else arch.w[j] < 0 for j in (i - 1, i + 1) if 0 <= j < nl]
            f[15] = sum(nb) / 2
        gain = ms0 / ms_new - ms0 / ms_cur
        cf = cost / BUDGET
        f[6] = 10 * gain; f[7] = max(-0.5, min(3.0, cf)); f[8] = min(2.0, wr); f[9] = rem
        f[12:16] = g_state[:3] + [f[15]]; f[16] = g_state[3]
        f[17] = max(-2.5, math.log10(max(gain, 1e-5) / max(cf, 0.01)) / 2)        # speed gained per share of budget
        rows.append(f); gains.append(gain)
    return torch.tensor(rows, dtype=torch.float32), gains


# ----------------------------------------------------------------------------------------------- one worker = one device
class Worker:
    def __init__(self, dev, P, sh):
        self.dev = torch.device(dev); self.cuda = self.dev.type == 'cuda'; self.sh = sh
        self.n32 = S.Net(P, dev, half=False, T=J.SEQ)
        self.n16 = S.Net(P, dev, half=True, T=J.SEQ) if self.cuda else None
        self.nS = self.n32                                     # the net search-time evaluations run on
        pool = J.split('search').to(self.dev); self.T = pool.shape[1]; self.n = len(pool) // SUBSETS
        self.ids = [pool[k * self.n:(k + 1) * self.n] for k in range(SUBSETS)]      # the rotating search subsets
        self.prompt = J.tokenizer()(J.PROMPTS[0])['input_ids']
        self.data = {}; self.base = {}; self.eval_s = 0.0; self.busy_s = 0.0

    @torch.no_grad()
    def tok_s(self, arch, runs, toks):
        dec = G.Decoder(S.materialize(self.n32, arch)).capture(); r = G.tok_per_s(dec, self.prompt, runs, toks)
        del dec
        return r

    @torch.no_grad()
    def load_split(self, name):
        self.data[name] = J.split(name).to(self.dev)
        self.base[name] = self.n32.seq_nll(self.data[name], self.n32.full())

    @torch.no_grad()
    def ratio(self, name, arch):
        """Perplexity of `arch` relative to the original on a split, in the architecture's own precision."""
        net = self.n16 if (arch.half and self.cuda) else self.n32
        d = (net.seq_nll(self.data[name], arch) - self.base[name]) / (self.T - 1)
        md = d.mean().item(); se = d.std().item() / math.sqrt(len(d))
        return dict(ratio=math.exp(md) if math.isfinite(md) else float('inf'), se_pct=round(100 * se, 3))

    @torch.no_grad()
    def make_root(self):
        """Residual stream of the original model entering every layer, per search subset (the prefix cache)."""
        net = self.nS; full = net.full(); self.root, self.root_b = [], []
        self.bases = [self.base['search'][k * self.n:(k + 1) * self.n] for k in range(SUBSETS)]
        for k in range(SUBSETS):
            acts = [net.embed(self.ids[k])]
            for i in range(net.NL): acts.append(net.layer(i, acts[-1], full))
            self.root.append(acts); self.root_b.append(self._bound(net.nll(acts[-1], self.ids[k]), k))

    def _bound(self, nll, k):
        d = (nll - self.bases[k]) / (self.T - 1)
        return d.mean().item(), d.std().item() / math.sqrt(len(d))

    @torch.no_grad()
    def eval_edit(self, new, i, acts, valid, k):
        """Score `new` on subset k. It differs from the architecture behind `acts` only at layer i, so recompute from
        the first layer that can have changed; everything before it comes from the cache."""
        t0 = time.perf_counter(); net = self.nS; s = min(i, valid); x = acts[s]; tmp = {}
        for j in range(s, net.NL):
            x = net.layer(j, x, new); tmp[j + 1] = x
        md, se = self._bound(net.nll(x, self.ids[k]), k); self.eval_s += time.perf_counter() - t0
        return md, se, tmp, s

    @torch.no_grad()
    def screen_edit(self, new, i, acts, valid, k):
        """Like eval_edit, on the first SCREEN_N sequences only. Returns (mean, standard error)."""
        t0 = time.perf_counter(); net = self.nS; s = min(i, valid); x = acts[s][:SCREEN_N]
        for j in range(s, net.NL): x = net.layer(j, x, new)
        d = (net.nll(x, self.ids[k][:SCREEN_N]) - self.bases[k][:SCREEN_N]) / (self.T - 1); self.eval_s += time.perf_counter() - t0
        return d.mean().item(), d.std().item() / math.sqrt(len(d))

    @torch.no_grad()
    def exact(self, arch, k):
        """Full measurement of an architecture on subset k (from scratch)."""
        return self._bound(self.nS.seq_nll(self.ids[k], arch), k)

    @torch.no_grad()
    def pool_bound(self, arch):
        """Mean and standard error of the NLL increase over the whole search text (all subsets)."""
        d = torch.cat([(self.nS.seq_nll(self.ids[k], arch) - self.bases[k]) / (self.T - 1) for k in range(SUBSETS)])
        return d.mean().item(), d.std().item() / math.sqrt(len(d))

    @torch.no_grad()
    def probe(self, edits):
        """Cost (mean increase of search NLL per token, subset 0) of single edits applied alone to the original model."""
        out = {}
        for name, arch, i in edits:
            md, se, _, _ = self.eval_edit(arch, i, self.root[0], self.nS.NL, 0); out[name] = md - self.root_b[0][0]
            self.sh.memo[(0, arch.key())] = (md, se)
        return out

    @torch.no_grad()
    def bench(self):
        """Seconds for one from-scratch scoring of an architecture on one subset: the layers, and the loss."""
        net = self.nS; full = net.full(); best = [1e9, 1e9]
        sync = (lambda: torch.cuda.synchronize(self.dev)) if self.cuda else (lambda: None)
        for _ in range(3):
            sync(); t0 = time.perf_counter(); x = net.embed(self.ids[0])
            for i in range(net.NL): x = net.layer(i, x, full)
            sync(); t1 = time.perf_counter(); net.nll(x, self.ids[0]); sync(); t2 = time.perf_counter()
            best = [min(best[0], t1 - t0), min(best[1], t2 - t1)]
        return dict(layers_s=round(best[0], 4), loss_s=round(best[1], 4))

    @torch.no_grad()
    def episode(self, pol, gen, k=0, greedy=False, chooser=None):
        t_ep = time.perf_counter()
        sh = self.sh; NL = self.nS.NL; arch = self.nS.full(); acts = list(self.root[k]); valid = NL
        md, se = self.root_b[k]; sc = 1.0; strikes = 0; blocked = set(); tr = []; acc = []; rej = []; used_shortcut = False
        while strikes < STRIKES and len(tr) < TMAX:
            cands = candidates(arch, blocked, sh.allow_half, sh.lat.nkv)
            if not cands: break
            F, gains = features(arch, cands, md + se, sc, strikes, sh)
            if chooser is not None: a = chooser(F); lp = 0.0
            else:
                ls = torch.log_softmax(pol(F), 0)
                a = int(ls.argmax()) if greedy else int(torch.multinomial(ls.exp(), 1, generator=gen))
                lp = float(ls[a])
            e = cands[a]; new = arch.apply(e); rem = 1 - (md + se) / BUDGET
            phi = [1.0, rem, rem * rem, strikes / STRIKES, 1.0 if arch.half else 0.0, sc - 1]
            computed = False
            if e[0] == 'half': nmd, nse = md, se               # search evaluations already run in this precision
            else:
                hit = sh.memo.get((k, new.key())); pred = md + edit_cost(arch, e, sh) if FAST else None
                if hit is not None: nmd, nse = hit; sh.hits += 1
                elif FAST and pred + se < PRED_LO * BUDGET:            # clearly inside the budget: take the prediction
                    nmd, nse = pred, se; sh.predicted += 1; used_shortcut = True
                elif FAST and pred > PRED_HI * BUDGET:                  # clearly over: refuse without measuring
                    nmd, nse = float('inf'), 0.0; sh.predicted += 1
                elif FAST and SCREEN_N < self.n and self._screen_clear(new, e[1], acts, valid, k):
                    nmd, nse = self._last; sh.screened += 1; used_shortcut = True
                else:
                    nmd, nse, tmp, s = self.eval_edit(new, e[1], acts, valid, k); sh.memo[(k, new.key())] = (nmd, nse)
                    computed = True; sh.evals += 1
                    for j in range(s + 1, e[1] + 1): acts[j] = tmp[j]
                    valid = max(valid, e[1])
            if math.isfinite(nmd) and nmd + nse <= BUDGET:
                r = 10 * gains[a]; arch = new; sc += gains[a]; md, se = nmd, nse; acc.append(e)
                if e[0] != 'half':
                    if computed:
                        for j in range(e[1] + 1, NL + 1): acts[j] = tmp[j]
                        valid = NL
                    else: valid = min(valid, e[1])
            else:
                r = -0.2; strikes += 1; blocked.add(e); rej.append(e)
            tr.append(dict(F=F, a=a, lp=lp, r=r, phi=phi))
        if FAST and acc and used_shortcut:                              # the end point is measured exactly if shortcuts were used
            md, se = self.exact(arch, k); sh.evals += 1
            if not (math.isfinite(md) and md + se <= BUDGET):
                tr[-1]['r'] -= 10 * (sc - 1) + 1.0; sh.overshoot += 1; sc = 1.0      # take back the speed it claimed
        self.busy_s += time.perf_counter() - t_ep
        return dict(tr=tr, arch=arch, S=sc, md=md, se=se, acc=acc, rej=rej, k=k)

    def _screen_clear(self, new, i, acts, valid, k):
        """Measure on the small slice; True if the answer is clearly on one side of the budget line."""
        m, s = self.screen_edit(new, i, acts, valid, k)
        if m + SCREEN_Z * s < 0.9 * BUDGET: self._last = (m, s); return True
        if m - SCREEN_Z * s > 1.1 * BUDGET: self._last = (float('inf'), 0.0); return True
        return False


def run_round(workers, pol, gens, n, counter):
    """n episodes shared between the workers (each takes the next one as soon as it is free). Every episode is
    scored on the next search subset in turn."""
    out = []; lock = threading.Lock(); left = [n]
    def loop(w, gen):
        while True:
            with lock:
                if left[0] <= 0: return
                left[0] -= 1; k = counter[0] % SUBSETS; counter[0] += 1
            ep = w.episode(pol, gen, k)
            with lock: out.append(ep)
    th = [threading.Thread(target=loop, args=(w, g)) for w, g in zip(workers, gens)]
    for t in th: t.start()
    for t in th: t.join()
    return out


def update(pol, opt, trans):
    """PPO-clip on this round's steps (all steps in one padded batch)."""
    N = len(trans); C = max(t['F'].shape[0] for t in trans)
    Fp = torch.zeros(N, C, FEAT); mask = torch.zeros(N, C, dtype=torch.bool)
    for n, t in enumerate(trans):
        c = t['F'].shape[0]; Fp[n, :c] = t['F']; mask[n, :c] = True
    act = torch.tensor([t['a'] for t in trans])[:, None]; lp0 = torch.tensor([t['lp'] for t in trans]); adv = torch.stack([t['adv'] for t in trans])
    for _ in range(EPOCHS):
        ls = torch.log_softmax(pol(Fp).masked_fill(~mask, -1e9), 1)
        ratio = torch.exp(ls.gather(1, act).squeeze(1) - lp0)
        ent = -(ls.exp() * ls).masked_fill(~mask, 0.0).sum(1)
        loss = (-torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv) - ENT * ent).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()


def main():
    torch.manual_seed(SEED)
    devs = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']
    say(event='setup', devices=[torch.cuda.get_device_name(i) for i in range(len(devs))] if devs[0] != 'cpu' else ['cpu'],
        loop_seconds=LOOP, search_seqs=J.SIZES['search'], search_subsets=SUBSETS, validation_seqs=J.SIZES['validation'],
        test_seqs=J.SIZES['test'], seq_len=J.SEQ)
    from transformers import AutoModelForCausalLM
    P = S.extract(AutoModelForCausalLM.from_pretrained(J.MODEL_DIR, dtype=torch.float32))
    sh = Shared(); workers = [Worker(d, P, sh) for d in devs]; w0 = workers[0]; NL, nkv = w0.n32.NL, w0.n32.nkv
    sh.NL, sh.nkv = NL, nkv

    # ---- usage statistics of the original model (calibration text) -> which neurons count as most-used
    st = w0.n32.stats(J.split('calib').to(w0.dev)); rank = [v.argsort(descending=True) for v in st['imp']]
    sh.aw, sh.mw, sh.kw = st['aw'].tolist(), st['mw'].tolist(), st['kw'].tolist()
    sh.mass = [[float(v[r[round(len(v) * wd):]].sum() / v.sum()) for wd in S.WIDTHS] for v, r in zip(st['imp'], rank)]
    for w in workers:
        w.n32.set_rank(rank)
        if w.n16 is not None: w.n16.set_rank(rank)
        w.load_split('search')
    for name in ('validation', 'test'): w0.load_split(name)
    base_ppl = {k: math.exp(v.sum().item() / (v.numel() * (w0.T - 1))) for k, v in w0.base.items()}
    say(event='baseline', ppl={k: round(v, 3) for k, v in base_ppl.items()})

    # ---- is 16-bit evaluation numerically safe on this model? then search-time scoring runs in 16-bit (faster)
    if w0.cuda:
        d0 = ((w0.n16.seq_nll(w0.data['search'], w0.n16.full()) - w0.base['search']) / (w0.T - 1)).mean().item()
        ok = math.isfinite(d0) and abs(d0) < 0.003; sh.allow_half = ok
        say(event='fp16_check', nll_shift_per_token=d0, usable=ok)
        if ok:
            for w in workers: w.nS = w.n16
    for w in workers: w.make_root()
    say(event='bench', one_scoring_from_scratch=w0.bench(), precision='fp16' if w0.nS is w0.n16 else 'fp32')

    # ---- latency model from real timings on this machine
    sh.lat, cal = calibrate(w0, [False, True] if sh.allow_half else [False])
    base_tps = 1000.0 / cal['fp32']['ms']['full']
    say(event='latency_model', base_tok_s=round(base_tps, 2), **cal)

    # ---- probe every part of the original model once: cost of removing it alone
    t = time.time(); full = w0.nS.full(); jobs = []
    for i in range(NL):
        jobs.append((('attn', i), full.apply(('attn', i)), i)); jobs.append((('mlp', i), full.apply(('mlp', i)), i))
        jobs.extend((('kv', i, g), full.apply(('kv', i, g)), i) for g in range(nkv))
        for wi in (1, 3, 5, 7):
            a = full.copy(); a.w[i] = wi; jobs.append((('w', i, wi), a, i))
    res = {}
    th = [threading.Thread(target=lambda w, js: res.update(w.probe(js)), args=(w, jobs[k::len(workers)])) for k, w in enumerate(workers)]
    for x in th: x.start()
    for x in th: x.join()
    sh.p_attn = [res[('attn', i)] for i in range(NL)]; sh.p_mlp = [res[('mlp', i)] for i in range(NL)]
    sh.p_kv = [[res[('kv', i, g)] for g in range(nkv)] for i in range(NL)]
    sh.p_w = []
    for i in range(NL):
        p = [0.0] * len(S.WIDTHS)
        for wi in (1, 3, 5, 7): p[wi] = res[('w', i, wi)]
        for wi in (2, 4, 6): p[wi] = (p[wi - 1] + p[wi + 1]) / 2
        sh.p_w.append(p)
    pct = lambda v: round(100 * (math.exp(v) - 1), 2)
    say(event='probe', seconds=round(time.time() - t, 1), evals=len(jobs),
        attn_removal_ppl_pct=[pct(v) for v in sh.p_attn], mlp_removal_ppl_pct=[pct(v) for v in sh.p_mlp],
        mlp_width90_ppl_pct=[pct(p[1]) for p in sh.p_w], mlp_width50_ppl_pct=[pct(p[5]) for p in sh.p_w],
        cheapest_kv_group_ppl_pct=[pct(min(v)) for v in sh.p_kv])

    # ---- the agent
    pol = Policy(); opt = torch.optim.Adam(pol.parameters(), lr=LR)
    gens = [torch.Generator().manual_seed(SEED * 1000 + k) for k in range(len(workers))]
    best = dict(S_pred=1.0, real=1.0, arch=None); gated = set()
    buf_phi, buf_G = [], []; hist = []; elog = open(f'{OUT}/episodes.jsonl', 'w'); n_ep = 0
    say(event='agent', policy_params=sum(p.numel() for p in pol.parameters()), edits_available=len(candidates(full, None, sh.allow_half, nkv)))

    def gate(ep, tag):
        """Whole search text, then validation, then real timing. Returns True if `ep` became the new best.
        Candidates are ranked by the latency model (it has no noise); the real timing has to confirm it."""
        nonlocal best
        key = ep['arch'].key() + (ep['arch'].half,)
        if key in gated or ep['S'] <= best['S_pred'] + 1e-4: return False
        gated.add(key); pmd, pse = w0.pool_bound(ep['arch'])
        if not (math.isfinite(pmd) and pmd + pse <= BUDGET):
            say(event='gate_fail', check='whole search text', where=tag, pred_speedup=round(ep['S'], 4), bound=round(math.exp(pmd + pse), 4)); return False
        v = w0.ratio('validation', ep['arch'])
        if v['ratio'] > TOL:
            say(event='gate_fail', check='validation', where=tag, pred_speedup=round(ep['S'], 4), val_ratio=round(v['ratio'], 4)); return False
        real = w0.tok_s(ep['arch'], CAL_RUNS, CAL_TOKENS) / base_tps
        if real < ep['S'] * (1 - TIMING_TOL):
            say(event='gate_fail', check='real speed', where=tag, pred_speedup=round(ep['S'], 4), real_speedup=round(real, 4)); return False
        best = dict(S_pred=ep['S'], real=real, arch=ep['arch'], val=v, search_bound=round(math.exp(pmd + pse), 4), edits=len(ep['acc']))
        json.dump(dict(pred_speedup=ep['S'], real_speedup=real, **ep['arch'].describe(nkv)), open(f'{OUT}/best_arch.json', 'w'))
        say(event='new_best', where=tag, real_speedup=round(real, 4), pred_speedup=round(ep['S'], 4), search_bound=best['search_bound'],
            val_ratio=round(v['ratio'], 4), edits=len(ep['acc']))
        return True

    def check(ep):
        """The same checks, for reporting only (does not change the saved best)."""
        pmd, pse = w0.pool_bound(ep['arch']); v = w0.ratio('validation', ep['arch'])
        ok = math.isfinite(pmd) and pmd + pse <= BUDGET and v['ratio'] <= TOL
        return dict(pred_speedup=round(ep['S'], 4), edits=len(ep['acc']), search_bound=round(math.exp(pmd + pse), 4), val_ratio=round(v['ratio'], 4),
                    passes=ok, real_speedup=round(w0.tok_s(ep['arch'], CAL_RUNS, CAL_TOKENS) / base_tps, 4) if ok else None)

    for w in workers: w.eval_s = w.busy_s = 0.0
    sh.evals = sh.hits = sh.predicted = sh.screened = sh.overshoot = 0
    t_loop = time.time(); rnd = 0; counter = [0]; pool_eps = []
    while time.time() - t_loop < LOOP:
        rnd += 1; snap = copy.deepcopy(pol); eps = run_round(workers, snap, gens, ROUND_EPISODES, counter)
        trans = []
        for ep in eps:
            n_ep += 1; G_ = 0.0
            for tr in reversed(ep['tr']):
                G_ += tr['r']; tr['G'] = G_
            trans.extend(ep['tr']); buf_phi.extend(tr['phi'] for tr in ep['tr']); buf_G.extend(tr['G'] for tr in ep['tr'])
            hist.append(ep['S'])
            elog.write(json.dumps(dict(ep=n_ep, round=rnd, subset=ep['k'], speedup=round(ep['S'], 4), bound=round(math.exp(ep['md'] + ep['se']), 4),
                                       accepted=ep['acc'], refused=ep['rej'])) + '\n')
        elog.flush()
        if trans:
            X = torch.tensor(buf_phi[-4000:]); y = torch.tensor(buf_G[-4000:])
            wv = torch.linalg.solve(X.T @ X + 1e-2 * torch.eye(X.shape[1]), X.T @ y)
            adv = torch.tensor([tr['G'] for tr in trans]) - torch.tensor([tr['phi'] for tr in trans]) @ wv
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            for tr, a in zip(trans, adv): tr['adv'] = a
            update(pol, opt, trans)
        pool_eps += eps
        if rnd % GATE_EVERY == 0:                                       # the save check is expensive: run it every few rounds
            for ep in sorted(pool_eps, key=lambda e: -e['S'])[:2]:
                if gate(ep, f'round {rnd}'): break
            pool_eps = []
        el = time.time() - t_loop
        say(event='round', n=rnd, episodes=n_ep, mean_speedup=round(statistics.mean(e['S'] for e in eps), 4),
            max_speedup=round(max(e['S'] for e in eps), 4), mean_edits=round(statistics.mean(len(e['acc']) for e in eps), 1),
            best_real=round(best['real'], 4), evals_per_s=round(sh.evals / el, 1), memo_hits=sh.hits, predicted=sh.predicted, screened=sh.screened,
            overshoot=sh.overshoot, decisions_per_s=round((sh.evals + sh.hits + sh.predicted + sh.screened) / el, 1), loop_s=round(el, 1))
    loop_s = time.time() - t_loop

    # ---- after learning: what the trained network does on its own (no sampling), one run per search subset,
    #      and a hand-written rule for reference (always take the edit with the best speed gained per budget spent)
    g0 = torch.Generator().manual_seed(SEED); greedy, hand = [], []
    for k in range(SUBSETS):
        ep = w0.episode(pol, g0, k, greedy=True); greedy.append(check(ep)); gate(ep, f'trained network, no sampling, subset {k}')
        hand.append(check(w0.episode(None, g0, k, chooser=lambda F: int(F[:, 17].argmax()))))
    k = max(1, len(hist) // 10)
    learn = dict(episodes=n_ep, rounds=rnd, loop_seconds=round(loop_s, 1), evals=sh.evals, memo_hits=sh.hits, predicted=sh.predicted, screened=sh.screened, overshoot=sh.overshoot,
                 worker_eval_seconds=[round(w.eval_s, 1) for w in workers], worker_busy_seconds=[round(w.busy_s, 1) for w in workers],
                 first_tenth_mean_speedup=round(statistics.mean(hist[:k]), 4), last_tenth_mean_speedup=round(statistics.mean(hist[-k:]), 4),
                 trained_network_no_sampling=greedy, hand_rule=hand)
    say(event='learning', **learn)
    torch.save(pol.state_dict(), f'{OUT}/policy.pt')

    # ---- final report on the saved best: unseen test text, real speed (alternating with the original), samples
    rep = dict(learning=learn, baseline=dict(ppl=base_ppl, tok_s=base_tps), latency_model=cal, devices=devs)
    if best['arch'] is not None:
        a = best['arch']; fulla = w0.n32.full(); tb, ta = [], []
        for _ in range(3):
            tb.append(w0.tok_s(fulla, CAL_RUNS, CAL_TOKENS)); ta.append(w0.tok_s(a, CAL_RUNS, CAL_TOKENS))
        tok = J.tokenizer(); M0 = S.materialize(w0.n32, fulla); M1 = S.materialize(w0.n32, a)
        d0, d1 = G.Decoder(M0).capture(), G.Decoder(M1).capture(); samples = []
        for pr in J.PROMPTS[:3]:
            ids = tok(pr)['input_ids']
            samples.append(dict(prompt=pr, original=tok.decode(d0.generate(ids, 40)), edited=tok.decode(d1.generate(ids, 40))))
        rep['best'] = dict(architecture=a.describe(nkv), edits=best['edits'],
                           params=S.n_params(M1), params_original=S.n_params(M0),
                           tok_s=round(statistics.median(ta), 2), tok_s_original=round(statistics.median(tb), 2),
                           real_speedup=round(statistics.median(ta) / statistics.median(tb), 4), predicted_speedup=round(best['S_pred'], 4),
                           search_bound=best['search_bound'], validation=w0.ratio('validation', a), test=w0.ratio('test', a),
                           search=w0.ratio('search', a), samples=samples)
        say(event='final', **{k: v for k, v in rep['best'].items() if k not in ('architecture', 'samples')})
        say(event='architecture', **{k: v for k, v in a.describe(nkv).items() if k not in ('attn', 'kv', 'w')})
        for s_ in samples: say(event='sample', **s_)
    else:
        say(event='final', note='no architecture passed the validation gate and beat the original speed')
    json.dump(rep, open(f'{OUT}/report.json', 'w'), indent=1)
    say(event='done')


if __name__ == '__main__': main()
