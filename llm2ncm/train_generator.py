#!/usr/bin/env python3
"""Stage B: train the NCM generator.

Backbone = the LLM itself with LoRA (full-attention models), so its language/code knowledge is reused;
models with recurrent / linear-attention layers (e.g. Qwen3.5) can't attend both ways inside a level, so
they get a small fresh transformer instead (the LLM still supplies embeddings, output layer, codes and
distillation targets). One forward pass trains every level: code CE per level + masked-token CE through
the LLM's own output layer (mask ratio ~ U(0,1), mask-predict style).

  python -I train_generator.py --model DIR --data work/data.pt --compressor work/compressor.pt --out work/generator
"""
import argparse, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from ncm_io import end_ids, load_compressor, load_llm, pick_device, save_generator
from ncm_model import NCMGenerator, level_attention_support

# Pool for the live "what does it answer now?" samples (--sample-every). Mixed on purpose: facts, arithmetic, code.
QUESTIONS = [
    "What is 2 + 2?", "What is the capital of France?", "Write a Python function that adds two numbers.",
    "What is 7 times 8?", "Who wrote Romeo and Juliet?", "Name three colors.", "What is the largest planet in our solar system?",
    "Write a Python function that returns the square of a number.", "How many days are in a week?",
    "What is the opposite of hot?", "Write a Python function that reverses a string.", "What is the chemical symbol for water?",
    "What is 100 divided by 4?", "Write a Python function that checks if a number is even.", "What language is spoken in Spain?",
    "Say hello in one sentence.", "What is the boiling point of water in Celsius?", "Write a Python loop that prints the numbers 1 to 5.",
    "What is the capital of Japan?", "Give me one tip for learning to code.",
]


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
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision (fp16 autocast + loss scaling on a GPU). A start-up check compares one batch against "
                         "fp32 and falls back to fp32 if the loss differs by more than 3% or is not finite")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="train for this many minutes: the learning-rate cosine follows the CLOCK instead of the step count and "
                         "training stops when the time is up (--steps is then only an upper bound)")
    ap.add_argument("--deadline-ts", type=float, default=0.0,
                    help="like --time-budget-min, but as an absolute unix time to finish by (at least 5 min are always allowed)")
    ap.add_argument("--log-every", type=int, default=10, help="print the training losses every N steps")
    ap.add_argument("--ar-first-level", action="store_true",
                    help="write the first (coarsest) code level one slot at a time instead of in parallel: a coherent outline "
                         "for a few extra passes")
    ap.add_argument("--soft-weight", type=float, default=0.0,
                    help="share of the token loss that is cross-entropy against the original model's top-k distribution "
                         "(needs data built with build_data.py --soft-k); 0 = hard targets only")
    ap.add_argument("--slot-emb", action="store_true",
                    help="learned per-slot embeddings for code-level and token slots (against neighbour repetition)")
    ap.add_argument("--mask-draws", type=int, default=1,
                    help="independent masked copies of each block per step; they share the prefix and codes, so this is "
                         "more token-loss signal per window for little extra compute (1 = the original single mask)")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--init-from", default="", help="continue from a saved generator folder (lora/ and extras.pt), e.g. an earlier run's generator/")
    ap.add_argument("--sample-every", type=int, default=100,
                    help="every N steps print the NCM's answers to a few random questions, to watch training live (0 = off)")
    ap.add_argument("--sample-n", type=int, default=3, help="questions per sample round")
    ap.add_argument("--sample-new", type=int, default=96, help="max new tokens per sampled answer")
    ap.add_argument("--val-windows", type=int, default=256, help="held-out windows for the validation loss (was 32: too noisy to compare runs)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = pick_device(a.device)
    tok, lm = load_llm(a.model, a.threads, dev)
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
    comp.to(dev)
    gen = NCMGenerator(lm, ccfg["d_lat"], ccfg["scales"], ccfg["L"], ccfg["codebook"], backbone=backbone, slot_emb=a.slot_emb, ar0=a.ar_first_level).to(dev)
    gen.train()
    if a.init_from:  # continue training from saved weights; the optimizer and learning-rate schedule start fresh
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        if backbone == "llm":
            res = set_peft_model_state_dict(lm, load_file(os.path.join(a.init_from, "lora", "adapter_model.safetensors")))
            assert not res.unexpected_keys, f"--init-from: LoRA keys do not match the model: {res.unexpected_keys[:3]}"
        missing = gen.load_state_dict(torch.load(os.path.join(a.init_from, "extras.pt"), map_location="cpu"), strict=False)
        bad = [k for k in missing.missing_keys if not k.startswith("lm.")]
        if bad:
            raise SystemExit(f"--init-from: generator extras missing keys: {bad[:5]}")
        print(f"continuing from {a.init_from}", flush=True)
    params = [p for p in gen.parameters() if p.requires_grad]
    print(f"backbone={backbone} trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

    d = torch.load(a.data, weights_only=False)
    prefix, block, codes = d["prefix"].long(), d["block"].long(), [c.long() for c in d["codes"]]
    pmask = d.get("prefix_mask")  # left-padded answer windows (build_data --prompt-frac); None for old data
    pm = lambda idx: None if pmask is None else pmask[idx].to(dev)
    s_ids, s_p = d.get("topk_ids"), d.get("topk_p")
    assert a.soft_weight <= 0 or s_ids is not None, "--soft-weight needs data built with build_data.py --soft-k"

    def sf(idx):  # soft targets of a batch: top-k ids and renormalised probabilities
        if a.soft_weight <= 0:
            return None
        p = s_p[idx].float()
        return s_ids[idx].long().to(dev), (p / p.sum(-1, keepdim=True)).to(dev)
    val = d["val_idx"]
    is_val = torch.zeros(len(block), dtype=torch.bool)
    is_val[val] = True
    tr = torch.nonzero(~is_val)[:, 0]
    vb = val[:a.val_windows]

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = 30
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    cfg = {"backbone": backbone, "backbone_note": why, "compressor": os.path.abspath(a.compressor), "model": a.model,
           "lora_r": a.lora_r if backbone == "llm" else None, "steps": a.steps, "batch": a.batch, "mask_draws": a.mask_draws, "slot_emb": a.slot_emb, "ar0": a.ar_first_level, "amp": a.amp, "init_from": a.init_from or None}

    @torch.no_grad()
    def val_loss():
        gen.eval()
        torch.manual_seed(1234)
        tot, n = {}, 0
        for i in range(0, len(vb), 64):
            c = vb[i:i + 64]
            _, parts = gen.loss(comp.rvq, prefix[c].to(dev), block[c].to(dev), [x[c].to(dev) for x in codes], corrupt=0.0,
                                level_weight=a.level_weight, prefix_mask=pm(c))
            for k, v in parts.items():
                tot[k] = tot.get(k, 0.0) + v * len(c)
            n += len(c)
        gen.train()
        torch.manual_seed(a.seed + int(time.time()))
        return {k: v / n for k, v in tot.items()}

    stops = end_ids(tok)
    q_rng = random.Random(a.seed)  # Python RNG only: sampling never touches (or depends on) the training RNG

    @torch.no_grad()
    def show_samples(step):
        gen.eval()
        print(f"--- step {step}: answers to {a.sample_n} random questions ---", flush=True)
        for q in q_rng.sample(QUESTIONS, a.sample_n):
            text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            out, _ = gen.generate(comp.rvq, tok(text, add_special_tokens=False)["input_ids"], a.sample_new, refine=4, end_ids=stops)
            ans = tok.decode(out, skip_special_tokens=True).strip()
            print(f"Q: {q}\nA: {ans[:400] if ans else '(empty) raw token ids: ' + str(out[:12])}", flush=True)
        gen.train()

    # mixed precision (opt-in), verified against fp32 on one batch before it is used
    use_amp = False
    amp_dev, amp_dtype = ("cuda", torch.float16) if dev == "cuda" else ("cpu", torch.bfloat16)
    if a.amp:
        try:
            bb = tr[:min(a.batch, 16)]
            chk = (comp.rvq, prefix[bb].to(dev), block[bb].to(dev), [c[bb].to(dev) for c in codes])
            gen.eval()
            with torch.no_grad():
                torch.manual_seed(1234)
                l32, _ = gen.loss(*chk, corrupt=0.0, prefix_mask=pm(bb))
                torch.manual_seed(1234)
                with torch.autocast(amp_dev, dtype=amp_dtype):
                    l16, _ = gen.loss(*chk, corrupt=0.0, prefix_mask=pm(bb))
            gen.train()
            use_amp = bool(torch.isfinite(l16)) and abs(float(l16) - float(l32)) < 0.03 * abs(float(l32)) + 1e-3
            print(f"amp check ({amp_dtype}): fp32 loss {float(l32):.4f} vs reduced precision {float(l16):.4f} -> "
                  f"{'USING reduced precision' if use_amp else 'FALLING BACK to fp32'}", flush=True)
        except Exception as e:  # never let the optional speed-up kill the run
            gen.train()
            print(f"amp check failed ({type(e).__name__}: {str(e)[:120]}) -> FALLING BACK to fp32", flush=True)
    scaler = torch.amp.GradScaler("cuda" if dev == "cuda" else "cpu", enabled=use_amp and dev == "cuda")
    budget = a.time_budget_min * 60.0 if a.time_budget_min else (max(a.deadline_ts - time.time(), 300.0) if a.deadline_ts else None)
    if budget:
        print(f"time budget: {budget / 60:.1f} min (learning rate follows the clock; training stops when it is used up)", flush=True)

    print("val before:", {k: round(v, 3) for k, v in val_loss().items()}, flush=True)
    best = float("inf")  # lowest validation loss so far: its checkpoint is kept separately (<out>_best) in case of late overfitting
    t0 = time.time()
    for step in range(1, a.steps + 1):
        if budget:
            frac = min((time.time() - t0) / budget, 1.0)
            for g_ in opt.param_groups:
                g_["lr"] = a.lr * min(1.0, step / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        b = tr[torch.randint(0, len(tr), (a.batch,))]
        with torch.autocast(amp_dev, dtype=amp_dtype, enabled=use_amp):
            loss, parts = gen.loss(comp.rvq, prefix[b].to(dev), block[b].to(dev), [c[b].to(dev) for c in codes],
                                   corrupt=a.corrupt, level_weight=a.level_weight, mask_draws=a.mask_draws, prefix_mask=pm(b),
                                   soft=sf(b), soft_weight=a.soft_weight)
        opt.zero_grad()
        scaler.scale(loss).backward()  # with the scaler disabled these three calls are exactly loss.backward() / opt.step()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if not budget:
            sched.step()
        last = step == a.steps or (bool(budget) and time.time() - t0 >= budget)
        if step % a.log_every == 0:
            tps = step * a.batch * (prefix.shape[1] + sum(ccfg["scales"]) + ccfg["L"] * a.mask_draws) / (time.time() - t0)
            print(f"step {step} " + " ".join(f"{k} {v:.3f}" for k, v in parts.items()) + f"  {tps:.0f} pos/s"
                  + (f"  {100 * min((time.time() - t0) / budget, 1.0):.0f}% of time" if budget else ""), flush=True)
        if step % a.save_every == 0 or last:
            vl = val_loss()
            print(f"  val step {step}:", {k: round(v, 3) for k, v in vl.items()}, flush=True)
            save_generator(gen, dict(cfg, trained_steps=step), a.out)
            if sum(vl.values()) < best:
                best = sum(vl.values())
                save_generator(gen, dict(cfg, trained_steps=step), a.out + "_best")
                print(f"  best validation loss so far (step {step}): checkpoint also saved to {a.out}_best", flush=True)
        if a.sample_every and (step % a.sample_every == 0 or last):
            show_samples(step)
        if last:
            break
    print(f"saved {a.out} after {step} steps ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
