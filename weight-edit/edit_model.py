#!/usr/bin/env python3
"""edit_model.py - improve SmolLM2-135M-Instruct by editing its weights directly. One command, no training.

    python edit_model.py --out smol-edited.gguf

What it does, in order (everything is closed form: averages, covariances and linear solves; no gradients):
  1. Re-encode   every weight to 4-bit with error-compensated rounding, so less is lost than with plain rounding.
  2. Read        record the model's last-layer activity on the example prompts in packs/ (behaviors.json, capitals.json).
  3. Write       behaviors: four hand-built neurons (direct yes/no answers decided by the model's own possible/impossible
                 signal, no "As an AI..." deflections, no carried-over yes/no from the previous answer, "you" not "I").
                 facts: every capital in packs/capitals.json, as one combined update of the last layer's output matrix
                 (each piece of each name is a key -> value pair; the update stays near zero on ordinary text).
  4. Save        round the edited last layer to 4-bit and write a GGUF file (same format Smol Chat and llama.cpp load).
  5. Report      before/after on held-out questions that the edit never read (packs/tests.json, test wordings of capitals).

Needs: torch, transformers, numpy, and llama.cpp's gguf-py (pass --llama-cpp, or pip install gguf).
Intermediate results are cached in --cache, so a rerun after a crash resumes where it stopped.
"""
import argparse, json, os, re, sys, time, random, hashlib, unicodedata
import numpy as np
import torch

F = torch.nn.functional
torch.set_grad_enabled(False)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
T0 = time.time()
LOGF = None

def log(*a):
    s = f'[{(time.time() - T0) / 60:5.1f} min] ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    if LOGF: LOGF.write(s + '\n'); LOGF.flush()

# =============================================================================================== 4-bit formats
def _f16(x): return x.to(torch.float16).to(torch.float32)

def q4_0(W):
    """Plain llama.cpp Q4_0 rounding (what the original 4-bit file contains)."""
    W = W.to(torch.bfloat16).to(torch.float32)
    o, i = W.shape; B = W.reshape(o, i // 32, 32)
    am = B.abs().argmax(-1, keepdim=True); mx = B.gather(-1, am)
    d = mx / -8.0; idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp((B * idv + 8.5).to(torch.int8).to(torch.float32), max=15)
    return ((q - 8.0) * _f16(d)).reshape(o, i)

def q8_0(W):
    W = W.to(torch.bfloat16).to(torch.float32)
    o, i = W.shape; B = W.reshape(o, i // 32, 32)
    d = B.abs().amax(-1, keepdim=True) / 127.0; idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    x = B * idv; q = torch.sign(x) * torch.floor(x.abs() + 0.5)
    return (q * _f16(d)).reshape(o, i)

def round_away(x): return torch.sign(x) * torch.floor(x.abs() + 0.5)

def find_scale(Wg, h):
    """Per-row scale for one block of 32: try the standard scale and smaller ones, both signs; keep the lowest
    input-weighted squared error. The value is exactly representable in f16."""
    m = Wg.abs().amax(1)
    best_e = torch.full_like(m, float('inf')); best_d = torch.zeros_like(m)
    for p in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        for sgn in (-1.0, 1.0):
            d = (sgn * p * m / 8.0).float().to(torch.float16).double()
            ds = torch.where(d == 0, torch.ones_like(d), d)
            code = torch.clamp(round_away(Wg / ds[:, None]) + 8, 0, 15)
            e = (((Wg - (code - 8) * d[:, None]) ** 2) * h[None, :]).sum(1)
            better = e < best_e
            best_e = torch.where(better, e, best_e); best_d = torch.where(better, d, best_d)
    return best_d

def gptq(W, H, percdamp=0.01, blocksize=128, group=32):
    """Error-compensated rounding to Q4_0 (GPTQ): each rounding error is pushed onto the weights not yet rounded,
    weighted by how the layer's inputs co-vary (H). Returns (dequantized weights, 4-bit codes, block scales)."""
    W = W.double().clone(); rows, cols = W.shape; H = H.double().clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(cols, dtype=H.dtype)
    hdiag = torch.diag(H).clone()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    CODES = torch.zeros((rows, cols), dtype=torch.uint8); D = torch.zeros((rows, cols // group), dtype=torch.float64)
    d = ds = None
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols); cnt = i2 - i1
        W1 = W[:, i1:i2].clone(); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            if col % group == 0:
                d = find_scale(W1[:, i:i + group], hdiag[col:col + group]); D[:, col // group] = d
                ds = torch.where(d == 0, torch.ones_like(d), d)
            w = W1[:, i]
            code = torch.clamp(round_away(w / ds) + 8, 0, 15)
            CODES[:, col] = code.to(torch.uint8)
            err = (w - (code - 8) * d) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return decode(CODES, D.float()), CODES, D.float()

def decode(CODES, D):
    rows, cols = CODES.shape
    return ((CODES.double() - 8).reshape(rows, cols // 32, 32) * D.double()[:, :, None]).reshape(rows, cols).float()

# =============================================================================================== model and data
MATS = [('attn_q', 'self_attn.q_proj', 'attn'), ('attn_k', 'self_attn.k_proj', 'attn'), ('attn_v', 'self_attn.v_proj', 'attn'),
        ('attn_output', 'self_attn.o_proj', 'o'), ('ffn_gate', 'mlp.gate_proj', 'mlp'), ('ffn_up', 'mlp.up_proj', 'mlp'), ('ffn_down', 'mlp.down_proj', 'down')]

def get(mod, path):
    for p in path.split('.'): mod = getattr(mod, p)
    return mod

class Stop(Exception): pass

class Lab:
    """The model, tokenizer and data, plus helpers to run, capture and generate."""
    def __init__(self, a):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
        self.FP = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.L = len(self.model.model.layers); self.LAST = self.L - 1
        self.IM_END = self.tok.convert_tokens_to_ids('<|im_end|>'); self.IMS = self.tok.convert_tokens_to_ids('<|im_start|>')
        cfg = self.model.config; self.n_head, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        chat = os.path.join(REPO, 'honesty-finetune', 'data', 'chatdata')
        self.B = json.load(open(os.path.join(HERE, 'packs', 'behaviors.json')))
        self.CAPS = json.load(open(os.path.join(HERE, 'packs', 'capitals.json')))
        self.TESTS = json.load(open(os.path.join(HERE, 'packs', 'tests.json')))
        rows1000 = json.load(open(os.path.join(chat, 'alpaca_1000.json')))
        rows200 = [r['row'] for r in json.load(open(os.path.join(chat, 'alpaca_200.json')))['rows']]
        qa = lambda r: (r['instruction'] + ('\n\n' + r['input'] if r.get('input') else ''), r['output'])
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)
        calib = [(self.ids(self.user(qa(r)[0])) + enc(qa(r)[1][:700]) + [self.IM_END])[:256] for r in rows1000[0:200] + rows1000[300:500]]
        calib += [(self.ids(c['messages']) + enc(c['good']) + [self.IM_END])[:256] for c in self.B['convs']]
        self.batches = self.make_batches(calib)
        self.dev_pairs = [(self.user(qa(r)[0]), qa(r)[1][:600]) for r in rows200[:100]]
        self.new_pairs = [(self.user(qa(r)[0]), qa(r)[1][:600]) for r in rows1000[500:600]]
        self.alp30 = [(self.user(qa(r)[0]), qa(r)[1][:500]) for r in rows200[:30]]

    @staticmethod
    def user(q): return [{'role': 'user', 'content': q}]
    def ids(self, messages, prefix=''):
        return self.tok.encode(self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + prefix, add_special_tokens=False)
    @staticmethod
    def pad_right(seqs):
        n = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), n), dtype=torch.long); att = torch.zeros_like(ids)
        for i, s in enumerate(seqs): ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        return ids, att
    def make_batches(self, seqs, bs=40):
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        return [self.pad_right([seqs[i] for i in order[s:s + bs]]) for s in range(0, len(order), bs)]

    # ---- weights
    def set_weights(self, codes, last_mlp=None):
        """Load a 4-bit model: q8 embeddings, decoded codes; the last MLP from `last_mlp` if given (else from codes or full precision)."""
        m = self.model
        with torch.no_grad():
            for n, p in m.named_parameters(): p.copy_(self.FP[n])
            m.model.embed_tokens.weight.copy_(q8_0(self.FP['model.embed_tokens.weight']))
            for key, (C, D) in codes.items():
                li, g = key.split('.'); get(m.model.layers[int(li)], dict((a, b) for a, b, _ in MATS)[g]).weight.copy_(decode(C, D))
            if last_mlp is not None:
                for k, v in last_mlp.items(): getattr(m.model.layers[self.LAST].mlp, k).weight.copy_(v)
    def set_plain_4bit(self):
        m = self.model
        with torch.no_grad():
            for n, p in m.named_parameters(): p.copy_(self.FP[n])
            m.model.embed_tokens.weight.copy_(q8_0(self.FP['model.embed_tokens.weight']))
            for L in m.model.layers:
                for _, path, _ in MATS: get(L, path).weight.copy_(q4_0(get(L, path).weight))
    def last_fp(self):
        return {k: self.FP[f'model.layers.{self.LAST}.mlp.{k}.weight'].clone() for k in ('gate_proj', 'up_proj', 'down_proj')}

    # ---- measurement
    def nll(self, pairs, micro=20):
        tot = 0.0
        for s0 in range(0, len(pairs), micro):
            seqs, spans = [], []
            for msgs, reply in pairs[s0:s0 + micro]:
                p = self.ids(msgs); r = self.tok.encode(reply, add_special_tokens=False) + [self.IM_END]
                seqs.append(p + r); spans.append((len(p), len(p) + len(r)))
            ids, att = self.pad_right(seqs)
            h = self.model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
            for i, (a, b) in enumerate(spans):
                lp = torch.log_softmax(self.model.lm_head(h[i, a - 1:b - 1]).float(), -1)
                tot += -lp.gather(-1, ids[i, a:b][:, None]).mean().item()
        return tot / len(pairs)

    def generate(self, convs, max_new=60, micro=64):
        reps = []
        for s0 in range(0, len(convs), micro):
            seqs = [self.ids(c) for c in convs[s0:s0 + micro]]; n = max(len(s) for s in seqs)
            pad = self.tok.pad_token_id or 0
            ids = torch.full((len(seqs), n), pad); att = torch.zeros_like(ids)
            for i, s in enumerate(seqs): ids[i, n - len(s):] = torch.tensor(s); att[i, n - len(s):] = 1
            out = self.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.1,
                                      pad_token_id=pad, eos_token_id=[self.IM_END, self.tok.eos_token_id])
            reps += [self.tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]
        return reps

    def capture_last(self, seqs, positions=None, micro=48):
        """Last layer: MLP input x and final pre-norm residual h. positions[i] = list of positions to keep (None = all)."""
        cur = {}; m = self.model
        h1 = m.model.layers[self.LAST].mlp.register_forward_pre_hook(lambda mod, inp: cur.__setitem__('x', inp[0].detach()))
        h2 = m.model.norm.register_forward_pre_hook(lambda mod, inp: cur.__setitem__('h', inp[0].detach()))
        X, Hh = [], []
        try:
            for s0 in range(0, len(seqs), micro):
                ch = seqs[s0:s0 + micro]; ids, att = self.pad_right(ch)
                m.model(input_ids=ids, attention_mask=att, use_cache=False)
                for i, s in enumerate(ch):
                    keep = list(range(len(s))) if positions is None else positions[s0 + i]
                    X.append(cur['x'][i, keep].clone()); Hh.append(cur['h'][i, keep].clone())
        finally:
            h1.remove(); h2.remove()
        return X, Hh

    def calib_last_inputs(self):
        cur = []; m = self.model
        def grab(mod, inp):
            cur.append(inp[0].detach().clone()); raise Stop()
        h = m.model.layers[self.LAST].mlp.register_forward_pre_hook(grab); xs = []
        try:
            for ids, att in self.batches:
                cur.clear()
                try: m.model(input_ids=ids, attention_mask=att, use_cache=False)
                except Stop: pass
                xs.append(cur[0][att.bool()].float())
        finally: h.remove()
        return torch.cat(xs)

    # ---- output side of the last layer
    def rms(self, h): return torch.sqrt((h * h).mean(-1) + 1e-5)
    def logits_from(self, h):
        gN = self.model.model.norm.weight.detach(); E = self.model.model.embed_tokens.weight.detach()
        return (h / self.rms(h)[:, None] * gN) @ E.T
    @staticmethod
    def acts(X, mats): return F.silu(X @ mats['gate_proj'].T) * (X @ mats['up_proj'].T)
    def mlp_out(self, X, mats): return self.acts(X, mats) @ mats['down_proj'].T
    def T(self, s): return self.tok.encode(s, add_special_tokens=False)[0]

# =============================================================================================== 1. re-encode
def reencode(lab):
    """Error-compensated 4-bit rounding of every attention and MLP matrix, layer by layer, each layer seeing inputs from
    the already-rounded layers before it. The last layer's MLP is left in full precision: the edits go there first."""
    m = lab.model
    with torch.no_grad():
        for n, p in m.named_parameters(): p.copy_(lab.FP[n])
        m.model.embed_tokens.weight.copy_(q8_0(lab.FP['model.embed_tokens.weight']))
    stream = []
    def grab(mod, args, kwargs):
        stream.append([args[0].detach().clone(), dict(kwargs)]); raise Stop()
    hk = m.model.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    try:
        for ids, att in lab.batches:
            try: m.model(input_ids=ids, attention_mask=att, use_cache=False)
            except Stop: pass
    finally: hk.remove()
    masks = [att.bool() for _, att in lab.batches]; ntok = int(sum(x.sum() for x in masks))
    run = lambda L, h, kw: (lambda o: o[0] if isinstance(o, tuple) else o)(L(h, **kw))
    codes = {}
    for li, L in enumerate(m.model.layers):
        t = time.time(); H = {}; cm = [None]
        def mk(key):
            def hook(mod, inp):
                x = inp[0][cm[0]].double(); hh = x.T @ x
                H[key] = hh if key not in H else H[key] + hh
            return hook
        hooks = [L.self_attn.q_proj.register_forward_pre_hook(mk('attn')), L.self_attn.o_proj.register_forward_pre_hook(mk('o')),
                 L.mlp.gate_proj.register_forward_pre_hook(mk('mlp')), L.mlp.down_proj.register_forward_pre_hook(mk('down'))]
        try:
            for (h, kw), msk in zip(stream, masks): cm[0] = msk; run(L, h, kw)
        finally:
            for x in hooks: x.remove()
        for name, path, key in MATS:
            if li == lab.LAST and name.startswith('ffn_'): continue
            mod = get(L, path); Q, C, D = gptq(mod.weight.detach().clone(), H[key] / ntok)
            mod.weight.copy_(Q); codes[f'{li}.{name}'] = (C, D)
        stream = [[run(L, h, kw), kw] for h, kw in stream]
        log(f'  re-encoded layer {li:2d} ({time.time() - t:.0f}s)')
    return codes

# =============================================================================================== 2-3. behaviors
class Behaviors:
    """Reads the model on the example prompts and builds four neurons for the last layer (closed form)."""
    COLN = 4.0
    PARAMS = dict(c_hd=30., alpha=8., theta=0., beta=1., c_you=25., k_o=5., k=8., t1=0.45, t2=0.65, k_p=5.)

    def __init__(self, lab):
        self.lab = lab; B = lab.B; L = lab
        S, meta = [], []                                   # meta: (kind, text, label, prev)
        for q, l in B['questions']: S.append(L.ids(L.user(q))); meta.append(('p3_yn', q, l, None))
        for q in B['open']: S.append(L.ids(L.user(q))); meta.append(('p3_open', q, None, None))
        for c in B['convs']: S.append(L.ids(c['messages'])); meta.append(('p3_conv', c['messages'][-1]['content'], c['label'], None))
        for msgs, rep in L.alp30: S.append(L.ids(msgs) + L.tok.encode(rep, add_special_tokens=False) + [L.IM_END]); meta.append(('p3_alpaca', '', None, None))
        for q, l in B['more_questions']: S.append(L.ids(L.user(q))); meta.append(('start_first_yn', q, l, None))
        for c in B['carry']: S.append(L.ids(c['messages'])); meta.append(('start_first_yn_conv', c['messages'][-1]['content'], c['label'], c['prev']))
        lab_ = {q: l for q, l in B['questions'] + B['more_questions'] if l}
        nq = [q for q, l in B['questions'] if l == 'N'] + [q for q, l in B['more_questions'] if l == 'N']
        yq = [q for q, l in B['questions'] if l == 'Y'] + [q for q, l in B['more_questions'] if l == 'Y']
        for q in nq: S.append(L.ids(L.user(q), 'No,')); meta.append(('comma_first', q, None, None))
        for q in yq: S.append(L.ids(L.user(q), 'Yes,')); meta.append(('comma_first', q, None, None))
        for c in B['convs']:
            if c['label'] == 'N': S.append(L.ids(c['messages'], 'No,')); meta.append(('comma_first', '', None, None))
        for c in B['carry']: S.append(L.ids(c['messages'], 'No,' if c['label'] == 'N' else 'Yes,')); meta.append(('comma_first', '', None, None))
        for q in B['about_assistant']: S.append(L.ids(L.user(q))); meta.append(('start_second', q, None, None))
        for pre in ('No,', 'Yes,'):
            for q in B['about_assistant_yesno']: S.append(L.ids(L.user(q), pre)); meta.append(('comma_second', q, None, None))
        for q, l in B['third_person']: S.append(L.ids(L.user(q))); meta.append(('start_third', q, l, None))
        for pre in ('No,', 'Yes,'):
            for q, _ in B['third_person']: S.append(L.ids(L.user(q), pre)); meta.append(('comma_third', q, None, None))
        self.seqs, self.meta = S, meta

    def read(self, X, H):
        """Tag every captured position and build the detectors, probe and read-out (all closed form)."""
        lab = self.lab; tok = lab.tok; S, meta = self.seqs, self.meta
        HDR = tok.encode('<|im_start|>assistant\n', add_special_tokens=False)
        FIRST = ('Can', 'Could', 'Should', 'Am', 'Is'); AMBIG = set(lab.B['ambiguous'])
        TAG, TXT, LAB, PREV, SEQ = [], [], [], [], []
        Xs, Hs = [], []
        for si, (s, (kind, text, label, prev)) in enumerate(zip(S, meta)):
            n = len(s)
            if kind.startswith('p3_'):
                roles, role, i = [], None, 0
                while i < n:
                    if s[i] == lab.IMS:
                        j = i + 1
                        while s[j] != 198: j += 1
                        role = tok.decode(s[i + 1:j]).strip(); roles += ['hdr'] * (j - i + 1); i = j + 1
                    else:
                        roles.append('hdr' if s[i] == lab.IM_END or (s[i] == 198 and s[i - 1] == lab.IM_END) else role); i += 1
                starts = {p for p in range(n) if p >= 3 and s[p - 3:p + 1] == HDR}
                for p in range(n):
                    if p in starts:
                        final = p == n - 1
                        if kind == 'p3_yn': t, l = ('start_first_yn' if text.split()[0] in FIRST else 'start_open'), label
                        elif kind == 'p3_open': t, l = 'start_open', None
                        elif kind == 'p3_alpaca': t, l = 'start_alpaca', None
                        elif final: t, l = ('start_first_yn_conv' if label else 'start_ambig' if text in AMBIG else 'start_conv'), label
                        else: t, l = 'start_conv_early', None
                    elif roles[p] == 'assistant': t, l = ('para' if tok.decode(s[p:p + 1]).endswith('\n') else 'reply'), None
                    else: t, l = 'other', None
                    TAG.append(t); TXT.append(text); LAB.append(l); PREV.append(None); SEQ.append((si, p))
                    Xs.append(X[si][p]); Hs.append(H[si][p])
            else:
                TAG.append(kind); TXT.append(text); LAB.append(label); PREV.append(prev); SEQ.append((si, n - 1))
                Xs.append(X[si][n - 1]); Hs.append(H[si][n - 1])
        self.X, self.H = torch.stack(Xs), torch.stack(Hs)
        self.TAG, self.TXT, self.LAB, self.PREV, self.SEQ = TAG, TXT, LAB, PREV, SEQ
        X, H = self.X, self.H
        sel = self.sel
        self.BASE = lab.last_fp(); self.H_MID = H - lab.mlp_out(X, self.BASE)
        ELSE = sel('reply', 'para', 'other')
        self.START = ['start_first_yn', 'start_first_yn_conv', 'start_open', 'start_conv', 'start_conv_early', 'start_alpaca', 'start_third', 'start_second']
        self.COMMAS = ['comma_first', 'comma_second', 'comma_third']
        self.FY = [i for i in sel('start_first_yn', 'start_first_yn_conv') if LAB[i]]
        mu_e = X[ELSE].mean(0); Xc = X[ELSE] - mu_e; SIG = Xc.T @ Xc / len(ELSE)
        SIG_INV = torch.linalg.inv(SIG + 0.1 * SIG.diagonal().mean() * torch.eye(SIG.shape[0]))
        mpara = X[sel('para')].mean(0)
        def lcmv(cons, also_zero_on_text=True):
            ms = [m for m, _ in cons] + ([mpara, mu_e] if also_zero_on_text else [])
            f = torch.tensor([float(v) for _, v in cons] + ([0.0, 0.0] if also_zero_on_text else []))
            C = torch.stack(ms).T
            return SIG_INV @ C @ torch.linalg.solve(C.T @ SIG_INV @ C, f)
        tm = lambda tags, v: [(X[sel(t)].mean(0), v) for t in tags]
        cmean = lambda tags: torch.stack([X[sel(t)].mean(0) for t in tags]).mean(0)
        NONSEC = [t for t in self.START if t != 'start_second']
        self.D_one = lcmv([(cmean(self.START), 1.0)] + tm(self.COMMAS, 0.0))
        self.D_user = lcmv([(cmean(NONSEC), 1.0)] + tm(['start_second'] + self.COMMAS, 0.0))
        self.D_pol = lcmv([(cmean(['start_first_yn', 'start_first_yn_conv']), 1.0)] +
                          tm(['start_open', 'start_conv', 'start_conv_early', 'start_alpaca', 'start_second', 'start_third'] + self.COMMAS, 0.0))
        self.D_pron = lcmv([(cmean(['comma_first']), 1.0)] + tm(['comma_second', 'comma_third'] + self.START, 0.0))
        ctx = lambda i: 'single' if TAG[i] == 'start_first_yn' else ('prev' + PREV[i]) if PREV[i] else 'conv'
        G = {}
        for i in self.FY: G.setdefault((LAB[i], ctx(i)), []).append(i)
        zeros = tm(['start_open', 'start_conv', 'start_conv_early', 'start_alpaca', 'start_second', 'start_third'], 0.0)
        self.PROBE = lcmv([(X[v].mean(0), 0.5 if k[0] == 'Y' else -0.5) for k, v in sorted(G.items())] + zeros, False)
        self.D_C = lcmv([(X[v].mean(0), 1.0) for k, v in sorted(G.items())] + zeros, False)
        T = lab.T
        self.YESF = [T(x) for x in ['Yes', 'Absol', 'Sure', 'Certainly']]; self.NO = T('No')
        lg = lab.logits_from(H[self.FY]); tb = torch.logsumexp(lg[:, self.YESF], -1) - lg[:, self.NO]
        tbd = {i: float(t) for i, t in zip(self.FY, tb)}
        self.READ = lcmv([(X[v].mean(0), sum(tbd[i] for i in v) / len(v)) for k, v in sorted(G.items())] + zeros, False)
        self.RBAR = float(lab.rms(H[self.FY]).mean())
        E = lab.model.model.embed_tokens.weight.detach(); gN = lab.model.model.norm.weight.detach()
        unit = lambda v: v / v.norm()
        HEDGE = [T(x) for x in ['As', 'I', 'While', 'Dear']]
        NEUTRAL = [T(x) for x in ['The', 'A', 'It', 'When', 'To', 'You', 'Here', 'For', 'In', 'If', 'There', 'This', 'First', 'One']]
        self.U_YN = gN * (E[self.YESF].mean(0) - E[self.NO]); self.yn_dir = unit(self.U_YN)
        lvl = unit(gN * (E[self.YESF + [self.NO]].mean(0) - E[NEUTRAL].mean(0)))
        hd = gN * (E[HEDGE].mean(0) - E[NEUTRAL].mean(0))
        for d in (self.yn_dir, unit(lvl - (lvl @ self.yn_dir) * self.yn_dir)): hd = hd - (hd @ d) * d
        self.hd_dir = unit(hd); self.you_dir = unit(gN * (E[T(' you')] - E[T(' I')]))

    def sel(self, *tags): return [i for i, t in enumerate(self.TAG) if t in tags]

    def pick_spare(self, k=4):
        """Least-used neurons of the last layer whose removal changes no reply-opening word on the example prompts."""
        lab = self.lab; X = self.X; BASE = self.BASE
        A = lab.acts(X, BASE); imp = A.abs().mean(0) * BASE['down_proj'].norm(dim=0)
        st = self.sel(*self.START); first = lab.logits_from(self.H[st]).argmax(-1)
        out = []
        for n in imp.argsort()[:24].tolist():
            m = {kk: v.clone() for kk, v in BASE.items()}; m['down_proj'][:, n] = 0
            if bool((lab.logits_from(self.H_MID[st] + lab.mlp_out(X[st], m)).argmax(-1) == first).all()): out.append(n)
            if len(out) == k: break
        return out

    def neurons(self):
        """(gate row, up row, down column) for: opener, yes/no polarity (+ and -, a soft step), pronoun."""
        P = self.PARAMS; C = self.COLN
        So = float(F.silu(torch.tensor(P['k_o']))); Sp = P['k'] * (P['t2'] - P['t1']); Sn = float(F.silu(torch.tensor(P['k_p'])))
        to_res = self.RBAR / float(self.U_YN.norm())
        v = to_res * (P['alpha'] * (self.PROBE - P['theta'] * self.D_C) - P['beta'] * self.READ)
        return [(P['k_o'] * self.D_user, self.D_one * (P['c_hd'] / So / C), -C * self.hd_dir),
                (P['k'] * (self.D_pol - P['t1'] * self.D_one), v / (Sp * C), C * self.yn_dir),
                (P['k'] * (self.D_pol - P['t2'] * self.D_one), v / (Sp * C), -C * self.yn_dir),
                (P['k_p'] * self.D_pron, self.D_pron * (P['c_you'] / Sn / C), C * self.you_dir)]

    def check(self, mats):
        """Exact dev check: first word of replies to the example yes/no questions."""
        lab = self.lab
        lg = lab.logits_from(self.H_MID[self.FY] + lab.mlp_out(self.X[self.FY], mats)); top = lg.argmax(-1).tolist()
        yes = set(self.YESF + [lab.T('Of')]); no = {self.NO}
        right = sum((self.LAB[i] == 'Y' and t in yes) or (self.LAB[i] == 'N' and t in no) for i, t in zip(self.FY, top))
        wrong = sum((self.LAB[i] == 'Y' and t in no) or (self.LAB[i] == 'N' and t in yes) for i, t in zip(self.FY, top))
        return right, wrong, len(self.FY)

# =============================================================================================== 3b. facts
def cap_first(c): return c[0].upper() + c[1:]

class Facts:
    """Each piece of each capital's name is a key -> value pair: key = the last layer's neuron activity right before
    that piece is written, value = a nudge towards that piece. All pairs are written into the last layer's output
    matrix at once, by one ridge-regularised least-squares solve that also keeps the output on ordinary text unchanged."""
    OTHER_PREFIX = ["People in {c} speak", "The largest city in {c} is", "The currency of {c} is the"]

    def __init__(self, lab):
        self.lab = lab; L = lab; tok = lab.tok
        seqs, rows = [], []      # rows: (seq index, position, target token or None, wording index, country index)
        enc = lambda s: tok.encode(s, add_special_tokens=False)
        def add(msgs, pre, answer, wi, ci):
            base = L.ids(msgs, pre); full = L.ids(msgs, pre + answer)
            if full[:len(base)] != base: return
            ans = full[len(base):]
            si = len(seqs); seqs.append(full)
            for j in range(len(ans)): rows.append((si, len(base) - 1 + j, ans[j], wi, ci))
        for ci, (c, cap) in enumerate(lab.CAPS['capitals']):
            for wi, w in enumerate(lab.CAPS['dev_ask']):
                msgs = L.user(w.format(c=c, C=cap_first(c)))
                add(msgs, f'The capital of {c} is', ' ' + cap, wi, ci)                   # "... is Canberra"
                add(msgs, '', cap, wi, ci)                                               # reply starts with the name
            for oi, w in enumerate(lab.CAPS['dev_other']):                              # same country, other questions: keep as is
                msgs = L.user(w.format(c=c)); pre = self.OTHER_PREFIX[oi].format(c=c)
                s1 = L.ids(msgs); si = len(seqs); seqs.append(s1); rows.append((si, len(s1) - 1, None, -1, ci))
                s2 = L.ids(msgs, pre); si = len(seqs); seqs.append(s2); rows.append((si, len(s2) - 1, None, -1, ci))
        self.seqs, self.rows = seqs, rows

    def solve(self, X, H, mats, exclude, xq_cal, preserve=None, margin=3.0, rounds=3, lam_grid=(0.03, 0.1, 0.3, 1.0)):
        lab = self.lab; rows = self.rows
        BASE = lab.last_fp()
        Hm = H - lab.mlp_out(X, BASE)                                 # residual before the last MLP
        keep = torch.ones(mats['down_proj'].shape[1], dtype=torch.bool); keep[exclude] = False
        K = lab.acts(X, mats)[:, keep].double()
        Kc = lab.acts(xq_cal, mats)[:, keep].double(); Ccal = Kc.T @ Kc
        if preserve is not None:                                       # other reply starts (example prompts): keep as is
            Xp, Hp = preserve
            Kp = lab.acts(Xp, mats)[:, keep].double()
        tgt = torch.tensor([t if t is not None else -1 for _, _, t, _, _ in rows])
        is_fact = tgt >= 0
        E = lab.model.model.embed_tokens.weight.detach().double(); gN = lab.model.model.norm.weight.detach().double()
        D0 = mats['down_proj'].double()

        def residual_needs(dW, idx):
            h = (Hm[idx] + lab.mlp_out(X[idx], mats)).double() + K[idx] @ dW.T
            lg = lab.logits_from(h.float()).double(); r = torch.zeros_like(h)
            for j, i in enumerate(idx.tolist()):
                if tgt[i] < 0: continue
                t = int(tgt[i]); l = lg[j].clone(); lt = float(l[t]); l[t] = -1e9; comp = int(l.argmax())
                need = margin - (lt - float(l[comp]))
                if need > 0:
                    u = gN * (E[t] - E[comp]); r[j] = need * float(lab.rms(h[j:j + 1].float())) * u / (u @ u)
            return r
        def fit(idx, lam):
            dW = torch.zeros(D0.shape[0], int(keep.sum()), dtype=torch.float64); R = torch.zeros(len(idx), D0.shape[0], dtype=torch.float64)
            Kf = K[idx]
            A = Kf.T @ Kf + lam * (torch.trace(Kf.T @ Kf) / torch.trace(Ccal)) * Ccal
            if preserve is not None: A = A + Kp.T @ Kp
            Ainv = torch.linalg.inv(A + 1e-6 * torch.trace(A) / A.shape[0] * torch.eye(A.shape[0], dtype=A.dtype))
            for _ in range(rounds):
                R = R + residual_needs(dW, idx)
                dW = (R.T @ Kf) @ Ainv
            return dW
        def accuracy(dW, idx):
            h = (Hm[idx] + lab.mlp_out(X[idx], mats)).double() + K[idx] @ dW.T
            top = lab.logits_from(h.float()).argmax(-1)
            f = is_fact[idx]
            return float((top[f] == tgt[idx][f]).float().mean())
        def cal_shift(dW):
            return float((Kc @ dW.T).norm(dim=1).mean() / (Kc @ D0[:, keep].T).norm(dim=1).mean())
        wi = torch.tensor([w for _, _, _, w, _ in rows])
        base_acc = accuracy(torch.zeros(D0.shape[0], int(keep.sum()), dtype=torch.float64), torch.arange(len(rows)))
        best = None
        for lam in lam_grid:                                           # pick the regularisation on a held-back dev wording
            accs = []
            for hold in range(len(lab.CAPS['dev_ask'])):
                tr = torch.nonzero((wi != hold)).flatten(); va = torch.nonzero(wi == hold).flatten()
                dW = fit(tr, lam); accs.append(accuracy(dW, va))
            a = sum(accs) / len(accs)
            log(f'    facts: regularisation {lam:<5}: accuracy on held-back dev wordings {a:.3f}')
            if best is None or a > best[0] + 0.005: best = (a, lam)
        dW = fit(torch.arange(len(rows)), best[1])
        log(f'    facts: next-piece accuracy on dev rows {base_acc:.3f} -> {accuracy(dW, torch.arange(len(rows))):.3f} | '
            f'chosen regularisation {best[1]} | change on ordinary text: {cal_shift(dW) * 100:.2f}% of the layer output')
        out = mats['down_proj'].clone(); out[:, keep] = (D0[:, keep] + dW).float()
        return out, dict(lam=best[1], heldback_acc=best[0], base_acc=base_acc)

# =============================================================================================== 4. bake and save
def bake(lab, mats, xq):
    """Round the edited last-layer MLP to 4-bit. gate/up: error-compensated rounding; down: re-fit so its output matches
    what the unrounded gate/up would give on the 4-bit model's own input, then error-compensated rounding."""
    x = xq.double(); N = len(x); Hx = x.T @ x / N
    Q, codes = {}, {}
    for k, name in (('gate_proj', 'ffn_gate'), ('up_proj', 'ffn_up')):
        Qk, C, D = gptq(mats[k], Hx); Q[k] = Qk; codes[f'{lab.LAST}.{name}'] = (C, D)
    act = lambda g, u: F.silu(x @ g.double().T) * (x @ u.double().T)
    a_q = act(Q['gate_proj'], Q['up_proj']); a_t = act(mats['gate_proj'], mats['up_proj'])
    Hd = a_q.T @ a_q / N; Cd = a_t.T @ a_q / N; Wd = mats['down_proj'].double()
    Hdd = Hd + 0.01 * torch.mean(torch.diag(Hd)) * torch.eye(Hd.shape[0], dtype=Hd.dtype)
    Wp = Wd + Wd @ (Cd - Hd) @ torch.linalg.inv(Hdd)
    Qd, C, D = gptq(Wp.float(), Hd); Q['down_proj'] = Qd; codes[f'{lab.LAST}.ffn_down'] = (C, D)
    return Q, codes

def import_gguf(llama_cpp):
    try:
        import gguf  # noqa
    except ImportError:
        sys.path.insert(0, os.path.join(llama_cpp, 'gguf-py'))
    import gguf
    from gguf.quants import dequantize
    return gguf, dequantize

def write_gguf(lab, codes, template, out, llama_cpp):
    """Copy a plain Q4_0 GGUF of the same model and overwrite each re-encoded tensor's 4-bit blocks. Rows of attn_q/attn_k
    are permuted the way llama.cpp's converter permutes them. Then read the file back and check every tensor."""
    gguf, dequantize = import_gguf(llama_cpp)
    buf = np.fromfile(template, dtype=np.uint8).copy()
    T = {t.name: t for t in gguf.GGUFReader(template).tensors}
    def perm(rows, n_head):
        return np.arange(rows).reshape(n_head, 2, rows // n_head // 2).swapaxes(1, 2).reshape(rows)
    expected = {}
    for key, (C, D) in codes.items():
        li, g = key.split('.'); name = f'blk.{li}.{g}.weight'; t = T[name]
        rows, cols = C.shape; nb = cols // 32
        assert t.tensor_type.name == 'Q4_0' and int(t.n_bytes) == rows * nb * 18, name
        p = perm(rows, lab.n_head) if g == 'attn_q' else perm(rows, lab.n_kv) if g == 'attn_k' else np.arange(rows)
        c = C.numpy()[p].reshape(rows, nb, 32); d = D.numpy()[p].astype(np.float16)
        qs = (c[:, :, :16] | (c[:, :, 16:] << 4)).astype(np.uint8)
        blocks = np.concatenate([d.reshape(rows, nb, 1).view(np.uint8).reshape(rows, nb, 2), qs], axis=-1)
        off = int(t.data_offset); buf[off:off + rows * nb * 18] = blocks.reshape(-1)
        expected[name] = decode(C, D).numpy()[p]
    buf.tofile(out)
    R = {t.name: t for t in gguf.GGUFReader(out).tensors}
    bad = [n for n, w in expected.items() if not np.array_equal(dequantize(R[n].data, R[n].tensor_type).astype(np.float32).reshape(w.shape), w)]
    # orientation check against the original weights: plain rounding of attn_q in the template must match HF rows after the same permutation
    w0 = dequantize(T['blk.0.attn_q.weight'].data, T['blk.0.attn_q.weight'].tensor_type).astype(np.float32)
    w1 = q4_0(lab.FP['model.layers.0.self_attn.q_proj.weight']).numpy()[perm(w0.shape[0], lab.n_head)]
    if not np.allclose(w0, w1, atol=1e-6): bad.append('row order of attn_q does not match the template')
    return bad

def build_template(lab, llama_cpp, cache):
    """A plain Q4_0 GGUF of the original model, made with llama.cpp (used as the container for the new weights)."""
    import subprocess
    t = os.path.join(cache, 'template-Q4_0.gguf')
    if os.path.exists(t): return t
    bf = os.path.join(cache, 'template-bf16.gguf')
    subprocess.run([sys.executable, os.path.join(llama_cpp, 'convert_hf_to_gguf.py'), lab_model_dir(lab), '--outtype', 'bf16', '--outfile', bf], check=True)
    subprocess.run([os.path.join(llama_cpp, 'build', 'bin', 'llama-quantize'), bf, t, 'Q4_0'], check=True)
    return t

def lab_model_dir(lab): return lab.model.config._name_or_path

# =============================================================================================== 5. report
def says(reply, names):
    n = lambda s: unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    r = n(reply)
    return any(re.search(r'(?<![a-z])' + re.escape(n(x)) + r'(?![a-z])', r) for x in names)

def deflects(t):
    return bool(re.search(r"as an ai|as a chatbot|as an artificial|as a language model|i'm not equipped|i am not equipped|i don't have the (capability|ability)|"
                          r"i don't have personal|i'm an ai|i am an ai|i'm not capable of|i can't physically|i cannot physically", t[:160], re.I))

def opener(label, t):
    t = t.strip(); yes = t.startswith(('Yes', 'Absolutely', 'Sure', 'Certainly', 'Of course')); no = t.startswith('No')
    return 'right' if (label == 'Y' and yes) or (label == 'N' and no) else 'wrong' if (yes or no) else 'none'

def report_replies(lab, cache, tag):
    path = os.path.join(cache, f'report_{tag}.json')
    if os.path.exists(path): return json.load(open(path))
    TS = lab.TESTS; C = lab.CAPS
    items = [('single', l, lab.user(q)) for q, l in TS['single']] + [('multi', None, m) for m in TS['multi']] + \
            [('about_assistant', None, lab.user(q)) for q in TS['about_assistant']] + [('third', l, lab.user(q)) for q, l in TS['third_person']] + \
            [('carry', c['label'], c['messages']) for c in TS['carry']] + [('showcase', None, TS['showcase'])]
    caps = []
    for c, cap in C['capitals']:
        for w in C['test_ask']: caps.append(('cap', cap, lab.user(w.format(c=c, C=cap_first(c)))))
        for a, b, q in C['test_multi']:
            caps.append(('cap_multi', cap, [{'role': 'user', 'content': a.format(c=c)}, {'role': 'assistant', 'content': b}, {'role': 'user', 'content': q}]))
        for w in C['test_other']: caps.append(('cap_other', cap, lab.user(w.format(c=c))))
    log(f'  report ({tag}): {len(items)} behavior prompts + {len(caps)} capital prompts')
    r1 = lab.generate([m for _, _, m in items], max_new=60)
    r2 = lab.generate([m for _, _, m in caps], max_new=24, micro=96)
    out = dict(items=[[k, l, rep] for (k, l, _), rep in zip(items, r1)], caps=[[k, cap, rep] for (k, cap, _), rep in zip(caps, r2)],
               nll=[lab.nll(lab.dev_pairs), lab.nll(lab.new_pairs)])
    json.dump(out, open(path, 'w'), indent=0)
    return out

def summarize(lab, R):
    al = lab.CAPS['aliases']; S = {}
    it = R['items']
    single = [opener(l, t) for k, l, t in it if k == 'single' and l in ('Y', 'N')]
    S['yes/no opened right (32)'] = single.count('right'); S['yes/no opened wrong (32)'] = single.count('wrong')
    S['carry-over right (16)'] = sum(opener(l, t) == 'right' for k, l, t in it if k == 'carry')
    S['third-person right (12)'] = sum(opener(l, t) == 'right' for k, l, t in it if k == 'third')
    S['"As an AI" deflections (114)'] = sum(deflects(t) for k, l, t in it if k != 'showcase')
    cap = [says(t, [c] + al.get(c, [])) for k, c, t in R['caps'] if k == 'cap']
    capm = [says(t, [c] + al.get(c, [])) for k, c, t in R['caps'] if k == 'cap_multi']
    oth = [says(t, [c] + al.get(c, [])) for k, c, t in R['caps'] if k == 'cap_other']
    S[f'capital named, new wordings ({len(cap)})'] = sum(cap)
    S[f'capital named, follow-up "capital there?" ({len(capm)})'] = sum(capm)
    S[f'capital named in language/biggest-city answers ({len(oth)})'] = sum(oth)
    S['general quality, dev text (lower is better)'] = round(R['nll'][0], 4)
    S['general quality, unseen text'] = round(R['nll'][1], 4)
    return S

# =============================================================================================== main
def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_model = os.path.join(REPO, 'model-surgery', 'runs', 'SmolLM2-135M-Instruct')
    ap.add_argument('--model', default=default_model if os.path.isdir(default_model) else 'HuggingFaceTB/SmolLM2-135M-Instruct')
    ap.add_argument('--out', default=os.path.join(HERE, 'smol-edited.gguf'))
    ap.add_argument('--template', default=os.path.join(REPO, 'model-surgery', 'runs', 'gguf', 'SmolLM2-135M-Instruct-Q4_0.gguf'),
                    help='plain Q4_0 GGUF of the same model (built with llama.cpp if missing)')
    ap.add_argument('--llama-cpp', default='/home/user/ggml-org/llama.cpp')
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    ap.add_argument('--no-facts', action='store_true'); ap.add_argument('--no-behaviors', action='store_true')
    ap.add_argument('--no-report', action='store_true')
    ap.add_argument('--threads', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.cache, exist_ok=True); LOGF = open(os.path.join(a.cache, 'edit_model.log'), 'a')
    log('loading', a.model)
    lab = Lab(a)
    cached = lambda name: os.path.join(a.cache, name)

    # 1. re-encode
    if os.path.exists(cached('reencoded.pt')): codes = torch.load(cached('reencoded.pt')); log('1. re-encode: cached')
    else:
        log('1. re-encode: error-compensated 4-bit rounding, layer by layer'); codes = reencode(lab); torch.save(codes, cached('reencoded.pt'))
    lab.set_weights(codes, last_mlp=lab.last_fp())

    # 2. read
    mats = lab.last_fp(); notes = {}
    if os.path.exists(cached('calib_last.pt')): xq = torch.load(cached('calib_last.pt'))
    else: xq = lab.calib_last_inputs(); torch.save(xq, cached('calib_last.pt'))
    bh = Behaviors(lab)
    log(f'2. read: last-layer activity on {len(bh.seqs)} example prompts')
    Xb, Hb = lab.capture_last(bh.seqs); bh.read(Xb, Hb)
    exclude = []
    if not a.no_behaviors:
        log('3. write behaviors: four neurons in the last layer')
        neur = bh.pick_spare(4); rows = bh.neurons()
        before = bh.check(mats)
        for n, (g, u, col) in zip(neur, rows):
            mats['gate_proj'][n] = g; mats['up_proj'][n] = u; mats['down_proj'][:, n] = col
        after = bh.check(mats)
        log(f'    neurons {neur} | example yes/no questions answered with the right first word: {before[0]}/{before[2]} -> {after[0]}/{after[2]} '
            f'(wrong {before[1]} -> {after[1]})')
        exclude = neur; notes['behavior_neurons'] = neur
    if not a.no_facts:
        fc = Facts(lab)
        log(f'3. write facts: {len(lab.CAPS["capitals"])} capitals, {sum(r[2] is not None for r in fc.rows)} key -> value pairs')
        pos = {}
        for si, p, *_ in fc.rows: pos.setdefault(si, []).append(p)
        Xs, Hs = lab.capture_last(fc.seqs, positions=[pos.get(i, [len(s) - 1]) for i, s in enumerate(fc.seqs)])
        # flatten in the same order as fc.rows
        flatX, flatH, ptr = [], [], {}
        for si, p, *_ in fc.rows:
            k = pos[si].index(p); flatX.append(Xs[si][k]); flatH.append(Hs[si][k])
        fX, fH = torch.stack(flatX), torch.stack(flatH)
        st = bh.sel(*bh.START)
        mats['down_proj'], notes['facts'] = fc.solve(fX, fH, mats, exclude, xq, preserve=(bh.X[st], bh.H[st]))

    # 4. bake and save
    log('4. save: round the edited last layer to 4-bit and write the GGUF')
    Q, last_codes = bake(lab, mats, xq)
    codes = dict(codes); codes.update(last_codes)
    torch.save({'codes': codes, 'notes': notes}, cached('final_codes.pt'))
    if not a.no_behaviors:
        lab_last = Q
        log(f'    after rounding: example yes/no questions right {bh.check(lab_last)[0]}/{len(bh.FY)}')
    template = a.template if os.path.exists(a.template) else build_template(lab, a.llama_cpp, a.cache)
    bad = write_gguf(lab, codes, template, a.out, a.llama_cpp)
    log(f'    wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB); read-back check: ' + ('all tensors exact' if not bad else f'PROBLEMS: {bad[:5]}'))
    if bad: sys.exit(1)

    # 5. report
    if a.no_report: return
    log('5. report: held-out questions, before (plain 4-bit) vs after')
    lab.set_plain_4bit(); Rb = report_replies(lab, a.cache, 'before')
    lab.set_weights(codes); Ra = report_replies(lab, a.cache, 'after_' + hashlib.sha1(open(a.out, 'rb').read()).hexdigest()[:8])
    Sb, Sa = summarize(lab, Rb), summarize(lab, Ra)
    w = max(len(k) for k in Sb)
    print('\n' + 'held-out check'.ljust(w) + '   before    after')
    for k in Sb: print(k.ljust(w) + f'   {Sb[k]!s:>6}   {Sa[k]!s:>6}')
    sb = [t for k, _, t in Rb['items'] if k == 'showcase'][0]; sa = [t for k, _, t in Ra['items'] if k == 'showcase'][0]
    print(f'\n"Can I collapse into a supernova?"\n  before: {sb[:150]!r}\n  after : {sa[:150]!r}')
    log('done')

if __name__ == '__main__':
    main()
