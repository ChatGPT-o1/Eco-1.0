#!/usr/bin/env python3
"""Train the EAGLE-style draft head on the original model's own answers (build_draft_data.py).

The original model stays frozen; only the head (~6M parameters) learns. --steps-ahead > 1 trains the multi-step
rollout ("training-time test", EAGLE-3) so drafts 2..k are trained on the inputs they really get when drafting.
With --time-budget-min the learning rate follows the clock and training stops when the time is up.
The saved head keeps the checkpoint with the best validation estimate of accepted drafts per round.

  python -I train_draft.py --model DIR --data draft_data.pt --out head.pt --time-budget-min 50 --device cuda --amp
"""
import argparse, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from eagle import EagleHead, draft_loss


def batch_of(seqs, resp, idx, max_len, pad, dev):
    rows = [seqs[i][:max_len] for i in idx]
    T = max(len(r) for r in rows)
    x = torch.full((len(rows), T), pad, dtype=torch.long)
    valid = torch.zeros(len(rows), T, dtype=torch.bool)
    for b, r in enumerate(rows):
        x[b, :len(r)] = torch.tensor(r)
        valid[b, :len(r)] = True
    return x.to(dev), valid.to(dev), torch.tensor([resp[i] for i in idx], device=dev)


def expected_accepted(st, steps):
    """Rough accepted-drafts-per-round estimate from the per-step accuracies: acc1 + acc1*acc2 + ..."""
    e, p = 0.0, 1.0
    for r in range(1, steps + 1):
        p *= st.get(f"acc{r}", 0.0)
        e += p
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000000, help="upper bound on optimizer steps")
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps-ahead", type=int, default=4, help="training-time-test rollout depth (1 = plain EAGLE-1 training)")
    ap.add_argument("--ce-weight", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.0, help="uniform noise added to input features (EAGLE-1 uses ~0.2)")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast + loss scaling on a GPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    head = EagleHead(lm.config).to(a.device)
    head.init_from(lm)
    print(f"draft head: {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M parameters; base model frozen", flush=True)

    d = torch.load(a.data, weights_only=False)
    seqs, resp = d["seqs"], d["resp_from"]
    nval = max(16, int(len(seqs) * a.val_frac))
    val, tr = list(range(nval)), list(range(nval, len(seqs)))
    use_amp = a.amp and a.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.0)
    budget = a.time_budget_min * 60.0 if a.time_budget_min else None
    warm = 100

    @torch.no_grad()
    def evaluate():
        head.eval()
        tot, n = {}, 0
        for i in range(0, len(val), a.batch):
            idx = val[i:i + a.batch]
            x, valid, rf = batch_of(seqs, resp, idx, a.max_len, pad, a.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                _, st = draft_loss(lm, head, x, valid, rf, a.ce_weight, 0.0, a.steps_ahead)
            for k, v in st.items():
                tot[k] = tot.get(k, 0.0) + v * len(idx)
            n += len(idx)
        head.train()
        res = {k: round(v / n, 4) for k, v in tot.items()}
        res["est_accepted"] = round(expected_accepted(res, a.steps_ahead), 3)
        return res

    best = -1.0
    t0 = time.time()
    head.train()
    for step in range(1, a.steps + 1):
        frac = min((time.time() - t0) / budget, 1.0) if budget else step / a.steps
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        x, valid, rf = batch_of(seqs, resp, rng.sample(tr, a.batch), a.max_len, pad, a.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss, st = draft_loss(lm, head, x, valid, rf, a.ce_weight, a.noise, a.steps_ahead)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        last = step == a.steps or (budget is not None and time.time() - t0 >= budget)
        if step % a.log_every == 0:
            print(f"step {step} loss {float(loss):.3f} " + " ".join(f"{k} {v:.3f}" for k, v in st.items())
                  + f"  {step / (time.time() - t0):.1f} steps/s" + (f"  {100 * frac:.0f}% of time" if budget else ""), flush=True)
        if step % a.eval_every == 0 or last:
            r = evaluate()
            print(f"  val step {step}: {r}", flush=True)
            if r["est_accepted"] > best:
                best = r["est_accepted"]
                torch.save({"state": head.state_dict(), "model": a.model, "step": step, "val": r,
                            "steps_ahead": a.steps_ahead}, a.out)
                print(f"  best so far -> saved {a.out}", flush=True)
        if last:
            break
    print(f"done after {step} steps ({time.time() - t0:.0f}s); best estimated accepted drafts per round {best:.3f}")


if __name__ == "__main__":
    main()
