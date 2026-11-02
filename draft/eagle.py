"""EAGLE-style draft head + exact speculative greedy decoding for a Hugging Face Llama-family model.

The draft head is one small transformer layer that sits beside the frozen original model. At each position it reads
the original model's final hidden state (the "feature") together with the embedding of the next token, and predicts
the NEXT feature; the original model's own output layer turns that predicted feature into a guessed token. Rolling it
forward k times drafts k tokens cheaply. The original model then checks all of them in ONE forward pass and keeps the
longest prefix that matches its own greedy choice, plus one token of its own. Nothing the draft head says reaches the
output unchecked, so the text is exactly what plain greedy decoding produces; a better head only means more tokens
per pass. (EAGLE: Li et al., 2024.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_theta(cfg):
    return getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 10000.0)


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope(x, pos, theta):
    """Rotary position embedding, Llama convention. x: (B, heads, T, head_dim); pos: (T,) absolute positions."""
    hd = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    ang = pos.float()[:, None] * inv[None]
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(x.dtype)
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(x.dtype)
    return x * cos + torch.cat([-x[..., hd // 2:], x[..., :hd // 2]], -1) * sin


class DraftLayer(nn.Module):
    """A Llama decoder layer (RMSNorm, grouped-query attention with RoPE, SwiGLU MLP) with a simple KV cache."""

    def __init__(self, H, heads, kv_heads, inter, theta, eps):
        super().__init__()
        self.h, self.kv, self.hd, self.theta = heads, kv_heads, H // heads, theta
        self.ln1, self.ln2 = RMSNorm(H, eps), RMSNorm(H, eps)
        self.q = nn.Linear(H, heads * self.hd, bias=False)
        self.k = nn.Linear(H, kv_heads * self.hd, bias=False)
        self.v = nn.Linear(H, kv_heads * self.hd, bias=False)
        self.o = nn.Linear(heads * self.hd, H, bias=False)
        self.gate = nn.Linear(H, inter, bias=False)
        self.up = nn.Linear(H, inter, bias=False)
        self.down = nn.Linear(inter, H, bias=False)

    def attend(self, x, pos, past_k, past_v, allow):
        """x: (B, T, H) at positions pos (T,); attends to the given past keys/values plus its own, under the boolean mask
        allow (T, S_past + T). Returns (output, this block's keys, values)."""
        B, T, _ = x.shape
        h = self.ln1(x)
        q = rope(self.q(h).view(B, T, self.h, self.hd).transpose(1, 2), pos, self.theta)
        k = rope(self.k(h).view(B, T, self.kv, self.hd).transpose(1, 2), pos, self.theta)
        v = self.v(h).view(B, T, self.kv, self.hd).transpose(1, 2)
        K, V = torch.cat(list(past_k) + [k], 2), torch.cat(list(past_v) + [v], 2)
        rep = self.h // self.kv
        a = F.scaled_dot_product_attention(q, K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1), attn_mask=allow[None, None])
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
        h = self.ln2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h)), k, v

    def forward(self, x, pos, cache=None):
        """Causal attention over the cache (dict with 'k'/'v', extended in place) and x. x: (B, T, H) at positions pos."""
        T = x.shape[1]
        has = cache is not None and cache.get("k") is not None
        past = cache["k"].shape[2] if has else 0
        allow = torch.arange(past + T, device=x.device)[None, :] <= (past + torch.arange(T, device=x.device))[:, None]
        out, k, v = self.attend(x, pos, [cache["k"]] if has else [], [cache["v"]] if has else [], allow)
        if cache is not None:
            cache["k"] = torch.cat([cache["k"], k], 2) if has else k
            cache["v"] = torch.cat([cache["v"], v], 2) if has else v
        return out


class EagleHead(nn.Module):
    """feature_t (B,T,H) + embedding of token_{t+1} (B,T,H) -> predicted feature_{t+1} (B,T,H)."""

    def __init__(self, cfg):
        super().__init__()
        H = cfg.hidden_size
        self.fc = nn.Linear(2 * H, H)
        self.layer = DraftLayer(H, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size,
                                rope_theta(cfg), cfg.rms_norm_eps)

    @torch.no_grad()
    def init_from(self, lm):
        """Start from a copy of the original model's LAST decoder layer: it already speaks the model's feature language."""
        last = lm.model.layers[-1]
        pairs = [(self.layer.q, last.self_attn.q_proj), (self.layer.k, last.self_attn.k_proj), (self.layer.v, last.self_attn.v_proj),
                 (self.layer.o, last.self_attn.o_proj), (self.layer.gate, last.mlp.gate_proj), (self.layer.up, last.mlp.up_proj),
                 (self.layer.down, last.mlp.down_proj), (self.layer.ln1, last.input_layernorm), (self.layer.ln2, last.post_attention_layernorm)]
        for mine, theirs in pairs:
            mine.weight.copy_(theirs.weight.float())

    def forward(self, feats, tok_emb, pos, cache=None):
        return self.layer(self.fc(torch.cat([feats, tok_emb.to(feats.dtype)], -1)), pos, cache)


# ------------------------------------------------------------------------------------------------- decoding
END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>", "<|end|>", "</s>")


def end_ids(tok):
    """Token ids that end an answer: eos plus the usual chat end markers that exist in this tokenizer."""
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for t in END_TOKENS:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    return ids


@torch.no_grad()
def greedy_plain(lm, prompt, max_new, eos):
    """Reference: ordinary greedy decoding with a KV cache, one forward pass per token. Returns (tokens, passes)."""
    dev = lm.lm_head.weight.device
    o = lm.model(input_ids=torch.tensor([prompt], device=dev), use_cache=True)
    cache, passes = o.past_key_values, 1
    tok = int(lm.lm_head(o.last_hidden_state[:, -1]).argmax(-1))
    out = [tok]
    while len(out) < max_new and tok not in eos:
        o = lm.model(input_ids=torch.tensor([[tok]], device=dev), past_key_values=cache, use_cache=True)
        cache, passes = o.past_key_values, passes + 1
        tok = int(lm.lm_head(o.last_hidden_state[:, -1]).argmax(-1))
        out.append(tok)
    return out, passes


@torch.no_grad()
def greedy_spec(lm, head, prompt, max_new, eos, k=4):
    """Exact speculative greedy decoding with the draft head drafting k tokens per round.
    Returns (tokens, verification passes including the prefill, list of accepted-draft counts per round).
    The tokens are identical to greedy_plain's (up to floating-point ties)."""
    dev = lm.lm_head.weight.device
    emb, lm_head = lm.model.embed_tokens, lm.lm_head
    n = len(prompt)
    o = lm.model(input_ids=torch.tensor([prompt], device=dev), use_cache=True)
    cache, passes = o.past_key_values, 1
    H = o.last_hidden_state
    t = int(lm_head(H[:, -1]).argmax(-1))
    out, accepted = [t], []
    if t in eos or max_new <= 1:
        return out, passes, accepted
    dc = {}
    f = head(H, emb(torch.tensor([prompt[1:] + [t]], device=dev)), torch.arange(n, device=dev), dc)
    last_f, L = f[:, -1:], n  # base and draft caches hold positions 0..L-1; t (pending) sits at position L
    while True:
        drafts, fc = [int(lm_head(last_f[:, -1]).argmax(-1))], last_f
        for i in range(1, k):
            fc = head(fc, emb(torch.tensor([[drafts[-1]]], device=dev)), torch.tensor([L + i - 1], device=dev), dc)
            drafts.append(int(lm_head(fc[:, -1]).argmax(-1)))
        o = lm.model(input_ids=torch.tensor([[t] + drafts], device=dev), past_key_values=cache, use_cache=True)
        cache, passes = o.past_key_values, passes + 1
        Hv = o.last_hidden_state
        g = lm_head(Hv[0]).argmax(-1).tolist()  # g[i] = the original model's greedy token for position L+i+1
        m = 0
        while m < k and drafts[m] == g[m]:
            m += 1
        new = drafts[:m] + [g[m]]
        accepted.append(m)
        if k > m:
            cache.crop(-(k - m))  # keep t, d1..dm (drop the rejected drafts); the correction token is fed next round
        dc["k"], dc["v"] = dc["k"][:, :, :L], dc["v"][:, :, :L]  # drop the speculative draft entries
        fe = head(Hv[:, :m + 1], emb(torch.tensor([new], device=dev)), torch.arange(L, L + m + 1, device=dev), dc)
        last_f = fe[:, -1:]
        for tok in new:
            out.append(tok)
            if tok in eos or len(out) >= max_new:
                return out, passes, accepted
        L, t = L + m + 1, new[-1]


def ttt_mask(T, r, device):
    """Attention mask for rollout step r (1 = true features) over the keys of steps 1..r, matching decoding exactly: the
    step-r query at position j belongs to the draft chain anchored at L = j - r + 2, so it sees the TRUE pairs before L
    (step-1 keys at positions <= j - r + 1) and, from each later step q, only the one key on its own chain (position
    j - (r - q)). Step 1 is plain causal attention."""
    j = torch.arange(T, device=device)[:, None]
    p = torch.arange(T, device=device)[None, :]
    blocks = [p <= j - r + 1] + [p == j - (r - q) for q in range(2, r + 1)]
    return torch.cat(blocks, 1)


def draft_loss(lm, head, x, valid, resp_from, ce_weight=0.1, noise=0.0, steps=1):
    """Training loss on a padded batch x (B, T) of token ids. valid: (B, T) bool; resp_from: (B,) index of the first
    response token (loss only where the target is a response token). steps > 1 adds EAGLE-3-style "training-time test":
    step r feeds the head its own step r-1 predictions (as drafting does) under ttt_mask, so drafts 2..k are trained
    for the inputs they will really get. Returns (loss, stats dict with per-step top-1 accuracy)."""
    with torch.no_grad():
        H = lm.model(input_ids=x, attention_mask=valid.long()).last_hidden_state.float()
    T = x.shape[1] - 1
    feats = H[:, :-1]
    if noise:
        feats = feats + noise * (torch.rand_like(feats) - 0.5)
    tok = lm.model.embed_tokens(x[:, 1:]).float()  # token x_{j+1} for pair j
    pos = torch.arange(T, device=x.device)
    is_resp = (torch.arange(x.shape[1], device=x.device)[None] >= resp_from[:, None]) & valid
    total, st, ks, vs, P = 0.0, {}, [], [], None
    for r in range(1, steps + 1):
        fin = feats if r == 1 else torch.cat([torch.zeros_like(P[:, :1]), P[:, :-1]], 1)  # step r: own previous predictions
        P, k, v = head.layer.attend(head.fc(torch.cat([fin, tok], -1)), pos, ks, vs, ttt_mask(T, r, x.device))
        ks.append(k)
        vs.append(v)
        has_anchor = (pos >= r - 1)[None]  # positions before r-1 have no draft chain at this depth
        m1 = is_resp[:, 1:] & has_anchor  # P[:, j] targets feature j+1
        reg = F.smooth_l1_loss(P[m1].float(), H[:, 1:][m1])
        m2 = is_resp[:, 2:] & has_anchor[:, :-1]  # lm_head(P[:, j]) targets token j+2
        logits = lm.lm_head(P[:, :-1][m2].to(lm.lm_head.weight.dtype)).float()
        tgt = x[:, 2:][m2]
        ce = F.cross_entropy(logits, tgt)
        total = total + reg + ce_weight * ce
        st[f"acc{r}"] = float((logits.argmax(-1) == tgt).float().mean())
        if r == 1:
            st["reg"], st["ce"] = float(reg.detach()), float(ce.detach())
    return total / steps, st
