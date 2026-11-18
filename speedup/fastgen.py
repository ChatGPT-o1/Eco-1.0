"""fastgen.py - lean greedy decoder for (edited) SmolLM models, with the whole decode step as one CUDA graph.

HF generate() launches ~15 small kernels per layer per token from Python; at batch 1 on a GPU that overhead, not the
math, sets the speed. Here one token step (embedding -> every layer -> lm_head -> argmax -> feed back) is captured once
and replayed, so a token is one graph launch. Works with any per-layer widths / head counts the edits produce.
"""
import torch, torch.nn.functional as F

class FastGen:
    def __init__(self, model, max_len=128):
        self.m = model; d = next(model.parameters()); self.dev, self.dt = d.device, d.dtype
        cfg = model.config; self.L = max_len; self.eps = cfg.rms_norm_eps
        hd = model.model.layers[0].self_attn.head_dim if model.model.layers else 64; self.hd = hd
        theta = getattr(cfg, 'rope_theta', None) or (getattr(cfg, 'rope_parameters', None) or {}).get('rope_theta', 10000.0)
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=self.dev, dtype=torch.float32) / hd))
        fr = torch.outer(torch.arange(max_len, device=self.dev, dtype=torch.float32), inv)
        emb = torch.cat([fr, fr], -1); self.cos, self.sin = emb.cos().to(self.dt), emb.sin().to(self.dt)
        self.layers = []
        for l in model.model.layers:
            at = l.self_attn; nkv = at.k_proj.out_features // hd; nq = at.q_proj.out_features // hd
            self.layers.append(dict(n1=l.input_layernorm.weight, n2=l.post_attention_layernorm.weight,
                q=at.q_proj.weight, k=at.k_proj.weight, v=at.v_proj.weight, o=at.o_proj.weight,
                g=l.mlp.gate_proj.weight, u=l.mlp.up_proj.weight, dn=l.mlp.down_proj.weight, nq=nq, nkv=nkv,
                kc=torch.zeros(nkv, max_len, hd, device=self.dev, dtype=self.dt),
                vc=torch.zeros(nkv, max_len, hd, device=self.dev, dtype=self.dt)))
        self.E = model.model.embed_tokens.weight; self.nf = model.model.norm.weight
        self.W_out = model.lm_head.weight
        self.tok = torch.zeros(1, dtype=torch.long, device=self.dev); self.pos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.ar = torch.arange(max_len, device=self.dev); self.graph = None

    def _norm(self, x, w): return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * w

    def _rope(self, x, c, s):
        h = x.shape[-1] // 2; rot = torch.cat([-x[..., h:], x[..., :h]], -1); return x * c + rot * s

    @torch.no_grad()
    def _step(self):
        """One greedy token: reads self.tok/self.pos, writes the next token into self.tok, advances self.pos."""
        x = self.E.index_select(0, self.tok)                                   # (1, d)
        c = self.cos.index_select(0, self.pos); s = self.sin.index_select(0, self.pos)
        mask = torch.where(self.ar <= self.pos, 0.0, float('-inf')).to(self.dt)  # (L,)
        for p in self.layers:
            h = self._norm(x, p['n1']); hd = self.hd
            q = (h @ p['q'].T).view(p['nq'], hd); k = (h @ p['k'].T).view(p['nkv'], hd); v = (h @ p['v'].T).view(p['nkv'], hd)
            q, k = self._rope(q, c, s), self._rope(k, c, s)
            p['kc'].index_copy_(1, self.pos, k.unsqueeze(1)); p['vc'].index_copy_(1, self.pos, v.unsqueeze(1))
            rep = p['nq'] // p['nkv']
            qg = q.view(p['nkv'], rep, hd)                                       # group query heads by their KV head
            att = (qg @ p['kc'].transpose(1, 2)) / hd ** 0.5 + mask               # (nkv, rep, L)
            o = (att.softmax(-1) @ p['vc']).reshape(1, -1)                        # (1, nq*hd)
            x = x + o @ p['o'].T
            h = self._norm(x, p['n2']); x = x + (F.silu(h @ p['g'].T) * (h @ p['u'].T)) @ p['dn'].T
        logits = self._norm(x, self.nf) @ self.W_out.T
        self.tok.copy_(logits.argmax(-1)); self.pos.add_(1)

    def _reset(self, ids):
        for p in self.layers: p['kc'].zero_(); p['vc'].zero_()
        self.pos.zero_()
        for t in ids[:-1]:                                                       # feed the prompt
            self.tok.fill_(t); self._step()
        self.tok.fill_(ids[-1])

    def capture(self):
        if self.dev.type != 'cuda': return
        self._reset([0, 0]); st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): self._step()                                       # warm-up allocations
        torch.cuda.current_stream().wait_stream(st)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph): self._step()

    def generate(self, ids, n):
        self._reset(list(ids)); return self.decode(n)

    def decode(self, n):
        out = []
        for _ in range(n):
            if self.graph is not None: self.graph.replay()
            else: self._step()
            out.append(self.tok.clone())            # stays on device; no sync per token
        return torch.cat(out).tolist()
