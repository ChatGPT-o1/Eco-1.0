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
    toks, _ = small.generate(comp.rvq, prompt, L, refine=2)
    assert len(toks) <= L
    print("5. small backbone loss + generation OK")
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
