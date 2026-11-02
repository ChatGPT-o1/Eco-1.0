"""Core modules for llm2ncm: turn a pretrained causal LM into a Next-Concept Model (NCM).

NCM (arXiv:2610.14162) represents a block of L tokens as levels of discrete codes, coarse to fine,
and generates one whole level per forward pass. Here the pretrained LLM is reused three ways:
  * encoder  - its frozen hidden states are compressed into the code levels (Compressor),
  * backbone - it is LoRA-tuned to predict each level in parallel (NCMGenerator),
  * teacher  - its own continuations are distillation targets, and it scores the NCM's outputs.

Layout of one generation block, after a causal prefix (prompt + earlier blocks):
    prefix | level-1 slots (s1) | ... | level-K slots (sK) | L token slots
Each slot attends to the prefix, all earlier levels, and its own level (both directions). Slot inputs
depend only on earlier levels, so a level's keys/values are computed once and cached for later levels.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FULL_ATTENTION_TYPES = {"full_attention", "sliding_attention", "chunked_attention"}
RECURRENT_MODEL_TYPES = {"mamba", "mamba2", "falcon_mamba", "rwkv", "jamba", "zamba", "zamba2", "bamba",
                         "qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "lfm2", "granitemoehybrid",
                         "nemotron_h", "falcon_h1", "minimax", "recurrent_gemma", "xlstm"}


def text_config(cfg):
    return getattr(cfg, "text_config", None) or cfg


def level_attention_support(cfg):
    """(ok, reason): can this backbone attend bidirectionally inside a level via a custom 4D mask?"""
    t = text_config(cfg)
    types = getattr(t, "layer_types", None)
    if types:
        other = sorted(set(types) - FULL_ATTENTION_TYPES)
        if other:
            return False, f"layer types {other} are causal/recurrent"
    mt = getattr(t, "model_type", "")
    if mt in RECURRENT_MODEL_TYPES:
        return False, f"model_type {mt!r} has recurrent layers"
    return True, "full attention"


def upsample(x, L):
    if x.shape[1] == L:
        return x
    return F.interpolate(x.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)


def downsample(x, s):
    if x.shape[1] == s:
        return x
    return F.adaptive_avg_pool1d(x.transpose(1, 2), s).transpose(1, 2)


class MultiScaleRVQ(nn.Module):
    """Multiscale residual VQ (VAR-style): level k pools the residual to scales[k] vectors and quantizes
    them with its own codebook (EMA updates, dead-code restart); the quantized vectors are upsampled
    back to L and subtracted from the residual."""

    def __init__(self, d, L, scales, codebook, beta=0.25, decay=0.99, dead=1e-3):
        super().__init__()
        self.d, self.L, self.scales, self.K = d, L, list(scales), codebook
        self.beta, self.decay, self.dead = beta, decay, dead
        n = len(scales)
        self.register_buffer("embed", torch.randn(n, codebook, d) * 0.02)
        self.register_buffer("ema_count", torch.zeros(n, codebook))
        self.register_buffer("ema_sum", torch.zeros(n, codebook, d))
        self.register_buffer("initted", torch.zeros(n, dtype=torch.bool))

    def nearest(self, k, x):
        cb = self.embed[k]
        d2 = x.pow(2).sum(-1, keepdim=True) - 2 * x @ cb.t() + cb.pow(2).sum(-1)
        return d2.argmin(-1)

    @torch.no_grad()
    def _ema(self, k, x, idx):
        if not self.initted[k]:  # initialise from data so no code starts dead
            sel = x[torch.randint(0, x.shape[0], (self.K,), device=x.device)]
            self.embed[k].copy_(sel + 0.01 * sel.std() * torch.randn_like(sel))
            self.ema_sum[k].copy_(self.embed[k])
            self.ema_count[k].fill_(1.0)
            self.initted[k] = True
            return
        cnt = torch.bincount(idx, minlength=self.K).to(x.dtype)
        s = torch.zeros_like(self.ema_sum[k]).index_add_(0, idx, x)
        self.ema_count[k].mul_(self.decay).add_(cnt, alpha=1 - self.decay)
        self.ema_sum[k].mul_(self.decay).add_(s, alpha=1 - self.decay)
        n = self.ema_count[k].sum()
        c = (self.ema_count[k] + 1e-5) / (n + self.K * 1e-5) * n
        self.embed[k].copy_(self.ema_sum[k] / c[:, None])
        dead = self.ema_count[k] < self.dead
        nd = int(dead.sum())
        if nd:
            sel = x[torch.randint(0, x.shape[0], (nd,), device=x.device)]
            self.embed[k][dead] = sel
            self.ema_sum[k][dead] = sel
            self.ema_count[k][dead] = 1.0

    def forward(self, z, n_levels=None, update=False):
        """z: (B, L, d) -> (reconstruction (B, L, d), codes [ (B, s_k) ], commitment loss)."""
        B, L, d = z.shape
        n = len(self.scales) if n_levels is None else n_levels
        residual, recon = z, torch.zeros_like(z)
        codes, commit = [], z.new_zeros(())
        for k in range(n):
            r = downsample(residual, self.scales[k])
            flat = r.reshape(-1, d).detach()
            if update and self.training:
                self._ema(k, flat, self.nearest(k, flat))
            idx = self.nearest(k, flat)
            q = self.embed[k][idx].view_as(r)
            commit = commit + F.mse_loss(r, q.detach())
            up = upsample(r + (q - r).detach(), L)  # straight-through
            recon = recon + up
            residual = residual - up
            codes.append(idx.view(B, -1))
        return recon, codes, self.beta * commit

    def recon_from_codes(self, codes, B=None, device=None):
        """Sum of upsampled codebook vectors for the given levels (empty list -> zeros)."""
        if not codes:
            return torch.zeros(B, self.L, self.d, device=device)
        out = 0
        for k, idx in enumerate(codes):
            out = out + upsample(self.embed[k][idx], self.L)
        return out

    def utilization(self, codes_k, k):
        return torch.unique(codes_k).numel() / self.K


class Compressor(nn.Module):
    """Frozen-LLM hidden states -> code levels -> (training only) token reconstruction."""

    def __init__(self, hidden, d, L, scales, codebook, dec_layers=2, heads=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(hidden, d), nn.LayerNorm(d))
        self.rvq = MultiScaleRVQ(d, L, scales, codebook)
        self.pos = nn.Parameter(torch.zeros(L, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(layer, dec_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden))

    def encode(self, h, n_levels=None, update=False):
        return self.rvq(self.inp(h), n_levels, update)

    def decode(self, recon):
        return self.out(self.dec(recon + self.pos))  # hidden-size vectors; apply the LLM's lm_head outside


class SmallBackbone(nn.Module):
    """Fallback generator for recurrent/linear-attention LLMs: a fresh bidirectional-capable transformer
    working in the LLM's embedding space (LLM embeddings in, LLM lm_head out). No KV cache."""

    def __init__(self, hidden, d=512, layers=6, heads=8, max_pos=8192):
        super().__init__()
        self.inp = nn.Linear(hidden, d)
        self.pos = nn.Embedding(max_pos, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden))

    def forward(self, inputs_embeds, mask4d, position_ids):
        x = self.inp(inputs_embeds) + self.pos(position_ids.clamp(max=self.pos.num_embeddings - 1))
        return self.out(self.tf(x, mask=mask4d[0, 0]))


def block_mask(P, sizes, dtype, device=None):
    """Additive 4D mask (1, 1, T, T): causal prefix; each slot group sees prefix + earlier groups + itself."""
    T = P + sum(sizes)
    allow = torch.zeros(T, T, dtype=torch.bool, device=device)
    allow[:P, :P] = torch.ones(P, P, dtype=torch.bool, device=device).tril()
    start = P
    for s in sizes:
        allow[start:start + s, :start + s] = True
        start += s
    m = torch.zeros(T, T, dtype=dtype, device=device)
    m.masked_fill_(~allow, torch.finfo(dtype).min)
    return m[None, None]


class NCMGenerator(nn.Module):
    def __init__(self, lm, d_lat, scales, L, codebook, backbone="llm"):
        super().__init__()
        self.lm = lm
        causal = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
        # references, not submodules (they already live inside self.lm)
        self._refs = (causal.get_input_embeddings(), getattr(causal, causal.base_model_prefix), causal.get_output_embeddings())
        hidden = self.embed.weight.shape[1]
        self.scales, self.L, self.K, self.backbone = list(scales), L, codebook, backbone
        self.small = SmallBackbone(hidden) if backbone == "small" else None
        self.code_in = nn.Linear(d_lat, hidden)
        emb_std = float(self.embed.weight.std())
        nn.init.normal_(self.code_in.weight, std=emb_std / math.sqrt(d_lat))
        nn.init.zeros_(self.code_in.bias)
        self.level_emb = nn.Parameter(torch.zeros(len(scales) + 1, hidden))
        self.mask_emb = nn.Parameter(self.embed.weight.detach().mean(0).clone())
        self.heads = nn.ModuleList([nn.Linear(hidden, codebook) for _ in scales])

    embed = property(lambda self: self._refs[0])
    decoder = property(lambda self: self._refs[1])
    lm_head = property(lambda self: self._refs[2])

    def extra_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("lm.")}

    # ---------------------------------------------------------------- layout helpers
    def level_positions(self, P, k):
        s = self.scales[k]
        return [P + int((i + 0.5) * self.L / s) for i in range(s)]

    def token_positions(self, P):
        return [P + j for j in range(self.L)]

    def level_input(self, rvq, k, codes, B, device):
        prev = rvq.recon_from_codes(codes[:k], B, device)
        return self.code_in(downsample(prev, self.scales[k])) + self.level_emb[k]

    def token_input(self, rvq, codes, toks, fixed):
        full = rvq.recon_from_codes(codes)
        tok = torch.where(fixed[..., None], self.embed(toks), self.mask_emb.to(full.dtype))
        return self.code_in(full) + self.level_emb[-1] + tok

    def run(self, x, mask4d, pos, past=None, use_cache=False):
        if self.small is not None:
            return self.small(x, mask4d, pos), None
        out = self.decoder(inputs_embeds=x, attention_mask=mask4d, position_ids=pos,
                           past_key_values=past, use_cache=use_cache)
        return out.last_hidden_state, out.past_key_values

    # ---------------------------------------------------------------- training
    def loss(self, rvq, prefix, block, codes, corrupt=0.1, level_weight=1.0):
        B, P = prefix.shape
        dev = prefix.device
        inp = codes  # inputs see corrupted earlier levels (robustness to own errors); targets stay clean
        if corrupt > 0:
            inp = [torch.where(torch.rand(c.shape, device=dev) < corrupt, torch.randint_like(c, self.K), c) for c in codes]
        ratio = torch.rand(B, 1, device=dev)
        masked = torch.rand(B, self.L, device=dev) < ratio
        masked[torch.arange(B, device=dev), torch.randint(0, self.L, (B,), device=dev)] = True
        parts = [self.embed(prefix)] + [self.level_input(rvq, k, inp, B, dev) for k in range(len(self.scales))]
        parts.append(self.token_input(rvq, inp, block, ~masked))
        x = torch.cat(parts, 1)
        sizes = self.scales + [self.L]
        mask = block_mask(P, sizes, x.dtype, dev).expand(B, -1, -1, -1)
        pos = list(range(P))
        for k in range(len(self.scales)):
            pos += self.level_positions(P, k)
        pos = torch.tensor(pos + self.token_positions(P), device=dev)[None].expand(B, -1)
        h, _ = self.run(x, mask, pos)
        losses, off = {}, P
        for k, s in enumerate(self.scales):
            logits = self.heads[k](h[:, off:off + s])
            losses[f"level{k + 1}"] = F.cross_entropy(logits.reshape(-1, self.K).float(), codes[k].reshape(-1))
            off += s
        tok_logits = self.lm_head(h[:, off:off + self.L][masked]).float()
        losses["tokens"] = F.cross_entropy(tok_logits, block[masked])
        total = level_weight * sum(v for k, v in losses.items() if k != "tokens") + losses["tokens"]
        return total, {k: float(v.detach()) for k, v in losses.items()}

    # ---------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, rvq, prompt, max_new, refine=4, temperature=0.0, end_ids=(), use_cache=None):
        """Block-by-block NCM decoding. Returns (new token ids, forward passes used).
        use_cache=False recomputes the whole sequence every pass (reference path for testing)."""
        dev = self.code_in.weight.device
        cached = (self.small is None) if use_cache is None else (use_cache and self.small is None)
        seq, out, passes = list(prompt), [], 0
        cache = None
        if cached:
            ids = torch.tensor([seq], device=dev)
            o = self.decoder(inputs_embeds=self.embed(ids), position_ids=torch.arange(len(seq), device=dev)[None], use_cache=True)
            cache, passes = o.past_key_values, 1
        dtype = self.code_in.weight.dtype
        while len(out) < max_new:
            P = len(seq)
            codes, groups = [], []
            for k, s in enumerate(self.scales):
                xk = self.level_input(rvq, k, codes, 1, dev)
                pos = torch.tensor(self.level_positions(P, k), device=dev)[None]
                if cached:
                    mask = torch.zeros(1, 1, s, cache.get_seq_length() + s, dtype=dtype, device=dev)
                    h, cache = self.run(xk, mask, pos, cache, use_cache=True)
                else:
                    h = self._full_pass(seq, groups + [(xk, pos)], dtype, dev)
                groups.append((xk, pos))
                codes.append(pick(self.heads[k](h[:, -s:]), temperature))
                passes += 1
            toks = torch.zeros(1, self.L, dtype=torch.long, device=dev)
            fixed = torch.zeros(1, self.L, dtype=torch.bool, device=dev)
            tpos = torch.tensor(self.token_positions(P), device=dev)[None]
            for r in range(refine):
                xt = self.token_input(rvq, codes, toks, fixed)
                if cached:
                    mask = torch.zeros(1, 1, self.L, cache.get_seq_length() + self.L, dtype=dtype, device=dev)
                    h, cache = self.run(xt, mask, tpos, cache, use_cache=True)
                    cache.crop(-self.L)
                else:
                    h = self._full_pass(seq, groups + [(xt, tpos)], dtype, dev)
                passes += 1
                logits = self.lm_head(h[:, -self.L:]).float()
                probs = torch.softmax(logits / max(temperature, 1e-6), -1) if temperature > 0 else torch.softmax(logits, -1)
                pred = torch.multinomial(probs[0], 1)[:, 0][None] if temperature > 0 else probs.argmax(-1)
                conf = probs.gather(-1, pred[..., None])[..., 0]
                pred = torch.where(fixed, toks, pred)
                conf = conf.masked_fill(fixed, float("inf"))
                n_fix = self.L if r == refine - 1 else max(int(fixed.sum()) + 1, math.ceil(self.L * (1 - math.cos(math.pi / 2 * (r + 1) / refine))))
                top = conf[0].topk(n_fix).indices
                fixed[0, top] = True
                toks[0, top] = pred[0, top]
            if cached:
                cache.crop(-sum(self.scales))
            block = toks[0].tolist()
            stop = next((i for i, t in enumerate(block) if t in end_ids), len(block))
            take = block[:min(stop, max_new - len(out))]
            out += take
            seq += take
            if stop < len(block) or len(out) >= max_new:
                break
            if cached:
                ids = torch.tensor([block], device=dev)
                o = self.decoder(inputs_embeds=self.embed(ids), position_ids=torch.arange(P, P + self.L, device=dev)[None],
                                 past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                passes += 1
        return out, passes

    def _full_pass(self, seq, groups, dtype, dev):
        P = len(seq)
        x = torch.cat([self.embed(torch.tensor([seq], device=dev))] + [g[0] for g in groups], 1)
        pos = torch.cat([torch.arange(P, device=dev)[None]] + [g[1] for g in groups], 1)
        mask = block_mask(P, [g[0].shape[1] for g in groups], dtype, dev)
        h, _ = self.run(x, mask, pos)
        return h


def pick(logits, temperature):
    if temperature > 0:
        p = torch.softmax(logits.float() / temperature, -1)
        return torch.multinomial(p.reshape(-1, p.shape[-1]), 1).view(p.shape[:-1])
    return logits.argmax(-1)
