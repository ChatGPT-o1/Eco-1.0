#!/usr/bin/env python3
"""Stage B: train the NCM generator.

Backbone = the LLM itself with LoRA (full-attention models), so its language/code knowledge is reused;
models with recurrent / linear-attention layers (e.g. Qwen3.5) can't attend both ways inside a level, so
they get a small fresh transformer instead (the LLM still supplies embeddings, output layer, codes and
distillation targets). One forward pass trains every level: code CE per level + masked-token CE through
the LLM's own output layer (mask ratio ~ U(0,1), mask-predict style).

  python -I train_generator.py --model DIR --data work/data.pt --compressor work/compressor.pt --out work/generator
"""
import argparse, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from ncm_io import load_compressor, load_llm, save_generator
from ncm_model import NCMGenerator, level_attention_support


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--compressor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="auto", choices=["auto", "llm", "small"])
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--corrupt", type=float, default=0.1, help="share of input codes randomised (robustness)")
    ap.add_argument("--level-weight", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok, lm = load_llm(a.model, a.threads)
    ok, why = level_attention_support(lm.config)
    backbone = a.backbone if a.backbone != "auto" else ("llm" if ok else "small")
    if backbone == "llm" and not ok:
        raise SystemExit(f"--backbone llm is impossible for this model: {why}")
    if backbone == "small":
        print(f"NOTE: using the small fallback generator ({why}); the LLM is encoder/embeddings/teacher only", flush=True)
    for p in lm.parameters():
        p.requires_grad_(False)
    if backbone == "llm":
        from peft import LoraConfig, get_peft_model
        lm = get_peft_model(lm, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, target_modules="all-linear"))
    comp, ccfg = load_compressor(a.compressor)
    gen = NCMGenerator(lm, ccfg["d_lat"], ccfg["scales"], ccfg["L"], ccfg["codebook"], backbone=backbone)
    gen.train()
    params = [p for p in gen.parameters() if p.requires_grad]
    print(f"backbone={backbone} trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

    d = torch.load(a.data, weights_only=False)
    prefix, block, codes = d["prefix"].long(), d["block"].long(), [c.long() for c in d["codes"]]
    val = d["val_idx"]
    is_val = torch.zeros(len(block), dtype=torch.bool)
    is_val[val] = True
    tr = torch.nonzero(~is_val)[:, 0]
    vb = val[:32]

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = 30
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    cfg = {"backbone": backbone, "backbone_note": why, "compressor": os.path.abspath(a.compressor), "model": a.model,
           "lora_r": a.lora_r if backbone == "llm" else None, "steps": a.steps, "batch": a.batch}

    @torch.no_grad()
    def val_loss():
        gen.eval()
        torch.manual_seed(1234)
        _, parts = gen.loss(comp.rvq, prefix[vb], block[vb], [c[vb] for c in codes], corrupt=0.0, level_weight=a.level_weight)
        gen.train()
        torch.manual_seed(a.seed + int(time.time()))
        return parts

    print("val before:", {k: round(v, 3) for k, v in val_loss().items()}, flush=True)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = tr[torch.randint(0, len(tr), (a.batch,))]
        loss, parts = gen.loss(comp.rvq, prefix[b], block[b], [c[b] for c in codes], corrupt=a.corrupt, level_weight=a.level_weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 10 == 0:
            tps = step * a.batch * (prefix.shape[1] + sum(ccfg["scales"]) + ccfg["L"]) / (time.time() - t0)
            print(f"step {step} " + " ".join(f"{k} {v:.3f}" for k, v in parts.items()) + f"  {tps:.0f} pos/s", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            print(f"  val step {step}:", {k: round(v, 3) for k, v in val_loss().items()}, flush=True)
            save_generator(gen, dict(cfg, trained_steps=step), a.out)
    print(f"saved {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
