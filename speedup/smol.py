"""smol.py - SmolLM-135M as plain tensors, plus an architecture description (Arch) of what is kept.

An Arch never changes a weight. It only says, for every layer:
    attn[i]   keep the attention sublayer?
    kv[i]     which key/value head groups are kept (bitmask; each group carries its 3 query heads)
    w[i]      MLP width as an index into WIDTHS (most-used neurons are kept); -1 = MLP sublayer removed
and, for the whole model:
    half      run the weights in 16-bit floats instead of 32-bit

Net.layer / Net.nll evaluate any Arch exactly (removed parts are masked or skipped), so the perplexity of an
edited model costs one forward pass and no model surgery. materialize() builds the physically smaller weights
that the decoder (fastgen.py) runs, which is what speed is measured on.
"""
import math, torch, torch.nn.functional as F

WIDTHS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def extract(hf):
    """Pull the weights out of a HuggingFace LlamaForCausalLM (fp32, CPU)."""
    c = hf.config
    theta = getattr(c, 'rope_theta', None) or (getattr(c, 'rope_parameters', None) or {}).get('rope_theta', 10000.0)
    assert hf.lm_head.weight is hf.model.embed_tokens.weight or torch.equal(hf.lm_head.weight, hf.model.embed_tokens.weight)
    P = dict(E=hf.model.embed_tokens.weight.detach().float().cpu(), nf=hf.model.norm.weight.detach().float().cpu(),
             eps=c.rms_norm_eps, theta=theta, nq=c.num_attention_heads, nkv=c.num_key_value_heads,
             hd=c.hidden_size // c.num_attention_heads, layers=[])
    for l in hf.model.layers:
        a, m = l.self_attn, l.mlp
        P['layers'].append({k: v.weight.detach().float().cpu() for k, v in dict(
            n1=l.input_layernorm, n2=l.post_attention_layernorm, q=a.q_proj, k=a.k_proj, v=a.v_proj, o=a.o_proj,
            g=m.gate_proj, u=m.up_proj, dn=m.down_proj).items()})
    return P


class Arch:
    __slots__ = ('attn', 'kv', 'w', 'half')

    def __init__(self, nl, nkv):
        self.attn = [True] * nl; self.kv = [(1 << nkv) - 1] * nl; self.w = [0] * nl; self.half = False

    def copy(self):
        a = Arch.__new__(Arch); a.attn = list(self.attn); a.kv = list(self.kv); a.w = list(self.w); a.half = self.half
        return a

    def key(self):                       # perplexity does not depend on `half` during the search
        return (tuple(self.attn), tuple(self.kv), tuple(self.w))

    def apply(self, e):
        a = self.copy(); k = e[0]
        if k == 'attn': a.attn[e[1]] = False
        elif k == 'kv': a.kv[e[1]] &= ~(1 << e[2])
        elif k == 'p10': a.w[e[1]] += 1
        elif k == 'p30': a.w[e[1]] += 3
        elif k == 'mlp': a.w[e[1]] = -1
        elif k == 'half': a.half = True
        else: raise ValueError(e)
        return a

    def describe(self, nkv=3):
        nl = len(self.attn); gone = lambda bits: [g for g in range(nkv) if not bits >> g & 1]
        return dict(precision='fp16' if self.half else 'fp32',
                    attention_removed=[i for i in range(nl) if not self.attn[i]],
                    mlp_removed=[i for i in range(nl) if self.w[i] < 0],
                    kv_groups_removed={str(i): gone(self.kv[i]) for i in range(nl) if self.attn[i] and gone(self.kv[i])},
                    mlp_width={str(i): WIDTHS[self.w[i]] for i in range(nl) if self.w[i] > 0},
                    attn=list(self.attn), kv=list(self.kv), w=list(self.w))

    @staticmethod
    def from_lists(attn, kv, w, half):
        a = Arch.__new__(Arch); a.attn = list(attn); a.kv = list(kv); a.w = list(w); a.half = bool(half); return a


class Net:
    """The model's tensors on one device in one precision. Never modified after construction."""

    def __init__(self, P, device, half=False, T=256):
        self.dev = torch.device(device); self.dt = torch.float16 if half else torch.float32
        mv = lambda t: t.to(device=self.dev, dtype=self.dt)
        self.E, self.nf = mv(P['E']), mv(P['nf'])
        self.L = [{k: mv(v) for k, v in l.items()} for l in P['layers']]
        self.eps, self.nq, self.nkv, self.hd, self.theta = P['eps'], P['nq'], P['nkv'], P['hd'], P['theta']
        self.rep = self.nq // self.nkv; self.NL = len(self.L); self.dff = self.L[0]['g'].shape[0]
        inv = 1.0 / (P['theta'] ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        fr = torch.outer(torch.arange(T, dtype=torch.float32), inv); emb = torch.cat([fr, fr], -1)
        self.cos, self.sin = mv(emb.cos()), mv(emb.sin())
        self.rank = None; self._nm = {}; self._hm = {}

    def full(self): return Arch(self.NL, self.nkv)

    def set_rank(self, rank): self.rank = [r.to(self.dev) for r in rank]; self._nm = {}

    def nmask(self, i, widx):
        m = self._nm.get((i, widx))
        if m is None:
            m = torch.zeros(self.dff, device=self.dev, dtype=self.dt)
            m[self.rank[i][:round(self.dff * WIDTHS[widx])]] = 1; self._nm[(i, widx)] = m
        return m

    def hmask(self, bits):
        m = self._hm.get(bits)
        if m is None:
            m = torch.tensor([float(bits >> (h // self.rep) & 1) for h in range(self.nq)], device=self.dev, dtype=self.dt).view(1, self.nq, 1, 1)
            self._hm[bits] = m
        return m

    def norm(self, x, w):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * w

    def rope(self, x):
        T = x.shape[-2]; c, s = self.cos[:T], self.sin[:T]; h = x.shape[-1] // 2
        return x * c + torch.cat([-x[..., h:], x[..., :h]], -1) * s

    def attn_heads(self, p, x):
        """Per-head attention outputs (B, nq, T, hd), before the output projection."""
        h = self.norm(x, p['n1']); B, T, _ = h.shape
        q = (h @ p['q'].T).view(B, T, self.nq, self.hd).transpose(1, 2)
        k = (h @ p['k'].T).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = (h @ p['v'].T).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        k, v = k.repeat_interleave(self.rep, 1), v.repeat_interleave(self.rep, 1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def layer(self, i, x, arch):
        p = self.L[i]
        if arch.attn[i]:
            o = self.attn_heads(p, x); B, _, T, _ = o.shape
            if arch.kv[i] != (1 << self.nkv) - 1: o = o * self.hmask(arch.kv[i])
            x = x + o.transpose(1, 2).reshape(B, T, self.nq * self.hd) @ p['o'].T
        if arch.w[i] >= 0:
            h = self.norm(x, p['n2'])
            a = F.silu(h @ p['g'].T) * (h @ p['u'].T)
            if arch.w[i] > 0: a = a * self.nmask(i, arch.w[i])
            x = x + a @ p['dn'].T
        return x

    def embed(self, ids): return self.E[ids]

    def logits(self, x): return self.norm(x, self.nf) @ self.E.T

    def nll(self, x, ids, chunk=8):
        """Per-sequence summed next-token negative log-likelihood (float32), from the final residual stream."""
        out = []
        for c in range(0, x.shape[0], chunk):
            lg = self.logits(x[c:c + chunk, :-1]).float()
            out.append((torch.logsumexp(lg, -1) - lg.gather(-1, ids[c:c + chunk, 1:, None]).squeeze(-1)).sum(1))
        return torch.cat(out)

    def seq_nll(self, ids, arch, batch=16):
        """Full forward pass of an architecture: per-sequence NLL."""
        out = []
        for b in range(0, ids.shape[0], batch):
            x = self.embed(ids[b:b + batch])
            for i in range(self.NL): x = self.layer(i, x, arch)
            out.append(self.nll(x, ids[b:b + batch]))
        return torch.cat(out)

    @torch.no_grad()
    def stats(self, ids, batch=8):
        """Usage statistics of every part of the original model on calibration text (not search/validation/test).
        imp[i]   per-neuron importance: mean |activation| x norm of its output weights   (to keep the most-used)
        aw[i]    how much the attention sublayer writes, relative to the residual stream (ratio of RMS)
        kw[i][g] the same for each key/value head group
        mw[i]    the same for the MLP sublayer
        """
        NL = self.NL; arch = self.full(); n = 0
        imp = [torch.zeros(self.dff, device=self.dev) for _ in range(NL)]
        aw = torch.zeros(NL); mw = torch.zeros(NL); kw = torch.zeros(NL, self.nkv)
        rms = lambda t: t.float().pow(2).mean(-1).sqrt()
        for b in range(0, ids.shape[0], batch):
            x = self.embed(ids[b:b + batch]); B, T, _ = x.shape; n += 1
            for i in range(NL):
                p = self.L[i]
                o = self.attn_heads(p, x)
                xr = _robust_rms(x)
                for g in range(self.nkv):
                    og = (o * self.hmask(1 << g)).transpose(1, 2).reshape(B, T, -1) @ p['o'].T
                    kw[i, g] += (rms(og) / xr).mean().item()
                ao = o.transpose(1, 2).reshape(B, T, -1) @ p['o'].T
                aw[i] += (rms(ao) / xr).mean().item(); x = x + ao
                h = self.norm(x, p['n2']); a = F.silu(h @ p['g'].T) * (h @ p['u'].T)
                imp[i] += a.float().abs().mean(dim=(0, 1))
                mo = a @ p['dn'].T; mw[i] += (rms(mo) / _robust_rms(x)).mean().item(); x = x + mo
        imp = [(v / n * self.L[i]['dn'].float().norm(dim=0)).cpu() for i, v in enumerate(imp)]
        return dict(imp=imp, aw=aw / n, kw=kw / n, mw=mw / n)


def _robust_rms(x, drop=4):
    """RMS of the residual stream ignoring its few huge-magnitude dimensions (they would swamp the ratio)."""
    s = x.float().pow(2); top = s.topk(drop, dim=-1).values.sum(-1)
    return ((s.sum(-1) - top) / (x.shape[-1] - drop)).sqrt()


def materialize(net32, arch, device=None, half=None):
    """Physically smaller weights for the decoder: only what `arch` keeps, in its precision."""
    device = torch.device(device) if device is not None else net32.dev
    dt = torch.float16 if (arch.half if half is None else half) else torch.float32
    mv = lambda t: t.to(device=device, dtype=dt).contiguous()
    hd, rep = net32.hd, net32.rep; layers = []
    for i, p in enumerate(net32.L):
        d = {}
        if arch.attn[i]:
            groups = [g for g in range(net32.nkv) if arch.kv[i] >> g & 1]
            qi = torch.cat([torch.arange(h * hd, (h + 1) * hd) for g in groups for h in range(g * rep, (g + 1) * rep)]).to(net32.dev)
            ki = torch.cat([torch.arange(g * hd, (g + 1) * hd) for g in groups]).to(net32.dev)
            d['attn'] = dict(n=mv(p['n1']), q=mv(p['q'][qi]), k=mv(p['k'][ki]), v=mv(p['v'][ki]), o=mv(p['o'][:, qi]), nkv=len(groups))
        if arch.w[i] >= 0:
            keep = net32.rank[i][:round(net32.dff * WIDTHS[arch.w[i]])].sort().values
            d['mlp'] = dict(n=mv(p['n2']), g=mv(p['g'][keep]), u=mv(p['u'][keep]), dn=mv(p['dn'][:, keep]))
        layers.append(d)
    return dict(E=mv(net32.E), nf=mv(net32.nf), layers=layers, eps=net32.eps, hd=hd, rep=rep, theta=net32.theta,
                device=device, dtype=dt)


def n_params(M):
    n = M['E'].numel() + M['nf'].numel()
    for d in M['layers']:
        for part in d.values(): n += sum(v.numel() for v in part.values() if torch.is_tensor(v))
    return n
