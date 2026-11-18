"""fastgen.py - lean greedy decoder for an edited SmolLM (the weights materialize() returns), and its timing.

On a GPU the whole token step (embedding -> every kept sublayer -> output -> argmax -> feed back) is captured once as a
CUDA graph and replayed, so a token is one launch with no Python in the loop. On CPU the same step runs directly.
Handles any architecture: removed attention or MLP sublayers, narrower MLPs, fewer head groups, fp32 or fp16.
"""
import time, torch, torch.nn.functional as F


class Decoder:
    def __init__(self, M, max_len=128):
        self.M = M; self.dev, self.dt = M['device'], M['dtype']; self.L = max_len; self.hd = M['hd']; self.eps = M['eps']
        inv = 1.0 / (M['theta'] ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        fr = torch.outer(torch.arange(max_len, dtype=torch.float32), inv); emb = torch.cat([fr, fr], -1)
        self.cos, self.sin = emb.cos().to(self.dev, self.dt), emb.sin().to(self.dev, self.dt)
        self.layers = []
        for d in M['layers']:
            e = {}
            if 'attn' in d:
                a = dict(d['attn']); nkv = a['nkv']
                a['kc'] = torch.zeros(nkv, max_len, self.hd, device=self.dev, dtype=self.dt)
                a['vc'] = torch.zeros(nkv, max_len, self.hd, device=self.dev, dtype=self.dt); e['attn'] = a
            if 'mlp' in d: e['mlp'] = d['mlp']
            if e: self.layers.append(e)
        self.tok = torch.zeros(1, dtype=torch.long, device=self.dev); self.pos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.ar = torch.arange(max_len, device=self.dev); self.graph = None

    def _norm(self, x, w): return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * w

    @staticmethod
    def _rope(x, c, s):
        h = x.shape[-1] // 2; return x * c + torch.cat([-x[..., h:], x[..., :h]], -1) * s

    @torch.no_grad()
    def _step(self):
        """One greedy token: reads self.tok / self.pos, writes the next token into self.tok, advances self.pos."""
        M, hd = self.M, self.hd
        x = M['E'].index_select(0, self.tok)
        c = self.cos.index_select(0, self.pos); s = self.sin.index_select(0, self.pos)
        mask = torch.where(self.ar <= self.pos, 0.0, float('-inf')).to(self.dt)
        for e in self.layers:
            a = e.get('attn')
            if a is not None:
                h = self._norm(x, a['n']); nkv = a['nkv']
                q = self._rope((h @ a['q'].T).view(-1, hd), c, s); k = self._rope((h @ a['k'].T).view(nkv, hd), c, s)
                v = (h @ a['v'].T).view(nkv, hd)
                a['kc'].index_copy_(1, self.pos, k.unsqueeze(1)); a['vc'].index_copy_(1, self.pos, v.unsqueeze(1))
                att = (q.view(nkv, -1, hd) @ a['kc'].transpose(1, 2)) / hd ** 0.5 + mask
                x = x + (att.softmax(-1) @ a['vc']).reshape(1, -1) @ a['o'].T
            m = e.get('mlp')
            if m is not None:
                h = self._norm(x, m['n']); x = x + (F.silu(h @ m['g'].T) * (h @ m['u'].T)) @ m['dn'].T
        self.tok.copy_((self._norm(x, M['nf']) @ M['E'].T).argmax(-1)); self.pos.add_(1)

    def reset(self, ids):
        for e in self.layers:
            if 'attn' in e: e['attn']['kc'].zero_(); e['attn']['vc'].zero_()
        self.pos.zero_()
        for t in ids[:-1]:
            self.tok.fill_(t); self._step()
        self.tok.fill_(ids[-1])

    def capture(self):
        if self.dev.type != 'cuda': return self
        with torch.cuda.device(self.dev):
            self.reset([0, 0]); st = torch.cuda.Stream(self.dev); st.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(st):
                for _ in range(3): self._step()
            torch.cuda.current_stream(self.dev).wait_stream(st)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph): self._step()
        return self

    def decode(self, n):
        out = []
        for _ in range(n):
            if self.graph is not None: self.graph.replay()
            else: self._step()
            out.append(self.tok.clone())
        return torch.cat(out).tolist()

    def generate(self, ids, n):
        self.reset(list(ids)); return self.decode(n)


def tok_per_s(dec, ids, runs=5, new_tokens=64):
    """Tokens per second: best of `runs` timed decodes of `new_tokens` tokens after the same prompt."""
    sync = (lambda: torch.cuda.synchronize(dec.dev)) if dec.dev.type == 'cuda' else (torch.mps.synchronize if dec.dev.type == 'mps' else (lambda: None))
    best = 1e9
    for _ in range(runs + 1):                         # first run is a warm-up
        dec.reset(ids)
        sync()
        t0 = time.perf_counter(); dec.decode(new_tokens)
        sync()
        best = min(best, time.perf_counter() - t0)
    return new_tokens / best
