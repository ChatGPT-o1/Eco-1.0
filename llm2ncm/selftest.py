#!/usr/bin/env python3
"""Mechanics checks for llm2ncm on a small model (no training data needed):
  1. block mask layout   2. RVQ shapes + gradients   3. generator overfits one batch
  4. cached generation == full recomputation (the KV-cache speed path is exact)
  5. small (fallback) backbone runs

  python -I selftest.py MODEL_DIR
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from ncm_model import Compressor, NCMGenerator, block_mask, level_attention_support


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    lm = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32)
    ok, why = level_attention_support(lm.config)
    print("backbone level attention:", ok, why)
    H = lm.get_input_embeddings().weight.shape[1]
    L, scales, K, P, B = 16, [2, 4], 64, 12, 4

    m = block_mask(2, [1, 2], torch.float32)[0, 0] == 0
    want = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(m, want), m
    print("1. block mask OK")

    comp = Compressor(H, 32, L, scales, K)
    comp.train()
    h = torch.randn(B, L, H, requires_grad=True)
    recon, codes, commit = comp.encode(h, update=True)
    recon, codes, commit = comp.encode(h, update=True)
    out = comp.decode(recon)
    (out.pow(2).mean() + commit).backward()
    assert recon.shape == (B, L, 32) and [c.shape for c in codes] == [(B, 2), (B, 4)] and h.grad is not None
    print("2. RVQ shapes + gradients OK; utilization", [round(torch.unique(c).numel() / K, 2) for c in codes])

    lm = get_peft_model(lm, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear"))
    gen = NCMGenerator(lm, 32, scales, L, K)
    prefix = torch.randint(100, 2000, (B, P))
    block = torch.randint(100, 2000, (B, L))
    params = [p for p in gen.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3)
    first = None
    for step in range(40):
        loss, parts = gen.loss(comp.rvq, prefix, block, codes, corrupt=0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or parts
    print("3. one-batch overfit:", {k: round(v, 2) for k, v in first.items()}, "->", {k: round(v, 2) for k, v in parts.items()})
    assert parts["tokens"] < first["tokens"] * 0.5, "generator failed to overfit a single batch"

    gen.eval()
    prompt = prefix[0].tolist()
    a, pa = gen.generate(comp.rvq, prompt, 2 * L, refine=3, use_cache=True)
    b, pb = gen.generate(comp.rvq, prompt, 2 * L, refine=3, use_cache=False)
    print(f"4. cached passes={pa} tokens={a[:8]}...  uncached passes={pb}")
    assert a == b, f"cached and uncached generation differ:\n{a}\n{b}"
    print("4. cached generation == full recomputation OK")

    small = NCMGenerator(lm, 32, scales, L, K, backbone="small")
    loss, _ = small.loss(comp.rvq, prefix, block, codes)
    loss.backward()
    loss2, _ = small.loss(comp.rvq, prefix, block, codes, mask_draws=2)
    loss2.backward()
    toks, _ = small.generate(comp.rvq, prompt, L, refine=2)
    assert len(toks) <= L
    print("5. small backbone loss + generation OK")

    # 6. several independent masked copies of the block per window (--mask-draws)
    m = block_mask(2, [1, 2, 2], torch.float32, parallel_tail=2)[0, 0] == 0
    want = torch.tensor([[1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1, 1]], dtype=torch.bool)
    assert torch.equal(m, want), m
    tp = lambda: gen.token_input(comp.rvq, codes, block, torch.rand(B, L) > 0.5)
    ta, tb, tc = tp(), tp(), tp()
    with torch.no_grad():
        h_ab, h_ac = gen.hidden(comp.rvq, prefix, codes, [ta, tb]), gen.hidden(comp.rvq, prefix, codes, [ta, tc])
        h_a = gen.hidden(comp.rvq, prefix, codes, [ta])
    end_a = P + sum(scales) + L
    scale = float(h_a.abs().max())
    # copy 1 must not see copy 2 at all (exactly equal); next to another copy it equals the single-copy layout up to
    # float noise from the different sequence length (relative to the activation scale, which grows during training)
    assert float((h_ab[:, :end_a] - h_ac[:, :end_a]).abs().max()) == 0.0, "copy 1 changed when copy 2 changed"
    assert float((h_ab[:, :end_a] - h_a).abs().max()) < 1e-4 * scale, "a copy behaves differently next to another copy"
    assert float((h_ab[:, end_a:] - h_ac[:, end_a:]).abs().max()) > 1e-2 * scale, "copy 2 ignores its own input"
    gen.train()
    loss3, parts3 = gen.loss(comp.rvq, prefix, block, codes, mask_draws=3)
    loss3.backward()
    print("6. parallel masked copies: mask OK, copies independent, 3-draw loss", {k: round(v, 2) for k, v in parts3.items()})

    # 7. left-padded prefixes (on-policy answer windows): with prefix_mask the padding must be invisible
    gen.eval()
    npad, c1 = 3, [c[:1] for c in codes]
    tp1 = gen.token_input(comp.rvq, c1, block[:1], torch.rand(1, L) > 0.5)
    padded = torch.cat([torch.full((1, npad), 7), prefix[:1]], 1)
    pmask = torch.cat([torch.zeros(1, npad, dtype=torch.bool), torch.ones(1, P, dtype=torch.bool)], 1)
    with torch.no_grad():
        h_plain = gen.hidden(comp.rvq, prefix[:1], c1, [tp1])
        h_pad = gen.hidden(comp.rvq, padded, c1, [tp1], prefix_mask=pmask)
    err = float((h_pad[:, npad:] - h_plain).abs().max())
    assert torch.isfinite(h_pad).all() and err < 1e-4 * float(h_plain.abs().max()), f"padding leaks into real positions ({err:.2e})"
    pm_b = torch.ones(B, P, dtype=torch.bool)
    pm_b[0, :5] = False
    gen.train()
    l7, _ = gen.loss(comp.rvq, prefix, block, codes, prefix_mask=pm_b)
    l7.backward()
    l7s, _ = small.loss(comp.rvq, prefix, block, codes, prefix_mask=pm_b)
    l7s.backward()
    assert torch.isfinite(l7) and torch.isfinite(l7s), "padded batch gives a non-finite loss"
    print(f"7. left padding invisible to real positions (max err {err:.1e}); padded loss finite (llm + small backbone)")

    # 8. per-slot embeddings (--slot-emb): zero init == plain layout; then trainable and cache-exact
    gs = NCMGenerator(lm, 32, scales, L, K, slot_emb=True)
    gs.load_state_dict(gen.extra_state(), strict=False)  # same code_in / level_emb / mask_emb / heads as gen
    gen.eval(); gs.eval()
    torch.manual_seed(5)
    l_plain, _ = gen.loss(comp.rvq, prefix, block, codes, corrupt=0.0)
    torch.manual_seed(5)
    l_slot, _ = gs.loss(comp.rvq, prefix, block, codes, corrupt=0.0)
    assert float(l_plain.detach()) == float(l_slot.detach()), (float(l_plain.detach()), float(l_slot.detach()))
    gs.train()
    l_slot.backward()
    assert gs.tok_slot.grad is not None and float(gs.tok_slot.grad.abs().sum()) > 0, "slot embeddings get no gradient"
    with torch.no_grad():
        gs.tok_slot.normal_(0, 0.05)
        for p_ in gs.lvl_slot:
            p_.normal_(0, 0.05)
    gs.eval()
    a8, _ = gs.generate(comp.rvq, prompt, 2 * L, refine=3, use_cache=True)
    b8, _ = gs.generate(comp.rvq, prompt, 2 * L, refine=3, use_cache=False)
    assert a8 == b8, "slot embeddings break cached generation"
    print("8. slot embeddings: zero init == plain layout, trainable, cached == recomputed")
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
