#!/usr/bin/env python3
"""Stage A: train the multiscale residual-VQ compressor on the frozen LLM's hidden states.

Loss = token cross-entropy of a small decoder (through the LLM's frozen output layer) reconstructing the
block from the code levels + VQ commitment. Nested dropout trains with only the first k levels half the
time, so coarse levels carry the most information. Then every window is encoded to codes for stage B.

  python -I train_compressor.py --model DIR --data work/data.pt --out work/compressor.pt --scales 2,8
"""
import argparse, json, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from ncm_io import load_llm, pick_device, save_compressor
from ncm_model import Compressor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", default="2,8", help="codes per level, coarse to fine (each < block length)")
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--d-lat", type=int, default=256)
    ap.add_argument("--quant", default="vq", choices=["vq", "fsq"],
                    help="vq = EMA vector quantiser (original); fsq = finite scalar quantisation (no codebook, every code usable)")
    ap.add_argument("--fsq-levels", default="8,5,5,5", help="values per FSQ scalar; a code has prod(levels) options (8,5,5,5 = 1000)")
    ap.add_argument("--pred-weight", type=float, default=0.0,
                    help="weight of the predictability (rate) term: penalise codes a context model cannot predict from the prefix "
                         "(fsq only; needs data built by the current build_data.py)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--init-compressor", default="",
                    help="load this trained compressor (its own architecture) instead of training one; use with --steps 0 to only encode the windows")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=1.0,
                    help="LR at the last step as a fraction of --lr, cosine decay after the 100-step warmup (1.0 = constant, the old behaviour)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="seed of the train/validation split (default: follows --seed). Pin it to compare runs with different "
                         "--seed on the same held-out windows")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(a.threads)
    dev = pick_device(a.device)
    d = torch.load(a.data, weights_only=False)
    L, H = d["cfg"]["L"], d["cfg"]["hidden"]
    scales = [int(s) for s in a.scales.split(",")]
    assert all(0 < s < L for s in scales) and scales == sorted(scales), "scales must increase and be < block length"
    _, lm = load_llm(a.model)
    head = lm.get_output_embeddings()
    W = head.weight.detach().float().clone().to(dev)
    bias = head.bias.detach().float().clone().to(dev) if getattr(head, "bias", None) is not None else None
    del lm, head

    N = len(d["block"])
    perm = torch.randperm(N) if a.split_seed is None else torch.randperm(N, generator=torch.Generator().manual_seed(a.split_seed))
    nval = max(16, int(N * a.val_frac))
    val, tr = perm[:nval], perm[nval:]
    hidden, block = d["hidden"], d["block"].long()
    levels = [int(v) for v in a.fsq_levels.split(",")] if a.quant == "fsq" else None
    if a.quant == "fsq":
        a.codebook = 1
        for v in levels:
            a.codebook *= v
    use_pred = a.quant == "fsq" and a.pred_weight > 0
    ctx = d.get("ctx")
    assert not use_pred or ctx is not None, "--pred-weight needs the prefix summary 'ctx': rebuild data with the current build_data.py"
    saved_cfg = None
    if a.init_compressor:  # the trained compressor as saved: no training, only the encoding below
        from ncm_io import load_compressor
        assert a.steps == 0, "--init-compressor only encodes: pass --steps 0"
        comp, saved_cfg = load_compressor(a.init_compressor)
        comp = comp.to(dev)
        levels, use_pred = saved_cfg.get("fsq_levels"), saved_cfg.get("pred", False)
    else:
        comp = Compressor(H, a.d_lat, L, scales, a.codebook, quant=a.quant, levels=levels, pred=use_pred).to(dev)
    opt = torch.optim.AdamW(comp.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * (a.lr_final + (1 - a.lr_final) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / max(a.steps, 1)))))

    def logits_of(recon):
        z = comp.decode(recon) @ W.t()
        return z + bias if bias is not None else z

    @torch.no_grad()
    def evaluate():
        comp.eval()
        res = {}
        for nl in range(1, len(scales) + 1):
            correct = total = 0
            used = [set() for _ in scales]
            for i in range(0, nval, 64):
                b = val[i:i + 64]
                recon, codes, _ = comp.encode(hidden[b].float().to(dev), nl)
                correct += int((logits_of(recon).argmax(-1).cpu() == block[b]).sum())
                total += block[b].numel()
                for k, c in enumerate(codes):
                    used[k].update(c.flatten().tolist())
            res[f"levels_{nl}_token_acc"] = round(correct / total, 4)
        res["codebook_use"] = [round(len(u) / a.codebook, 3) for u in used]
        if use_pred:  # how predictable are the codes from the prefix? (nats per scalar; uniform guessing = ln(levels))
            r_sum = r_n = 0
            for i in range(0, min(nval, 256), 64):
                b = val[i:i + 64]
                comp.encode(hidden[b].float().to(dev))
                r_sum += float(comp.rate_loss(ctx[b].float().to(dev))) * len(b)
                r_n += len(b)
            res["pred_rate_nats"] = round(r_sum / r_n, 3)
            res["uniform_nats"] = round(sum(math.log(v) for v in levels) / len(levels), 3)
        comp.train()
        return res

    t0 = time.time()
    comp.train()
    hist = []
    for step in range(1, a.steps + 1):
        b = tr[torch.randint(0, len(tr), (a.batch,))]
        nl = len(scales) if random.random() < 0.5 else random.randint(1, len(scales))
        recon, codes, commit = comp.encode(hidden[b].float().to(dev), nl, update=True)
        ce = F.cross_entropy(logits_of(recon).reshape(-1, W.shape[0]), block[b].reshape(-1).to(dev))
        loss = ce + commit
        rate = None
        if use_pred:
            rate = comp.rate_loss(ctx[b].float().to(dev), nl)
            loss = loss + a.pred_weight * rate
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(comp.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"step {step} ce {ce.item():.3f} commit {commit.item():.4f}" + (f" rate {rate.item():.3f}" if rate is not None else "")
                  + f" levels {nl} ({time.time() - t0:.0f}s)", flush=True)
        if step % 500 == 0 or step == a.steps:
            r = evaluate()
            r["step"] = step
            hist.append(r)
            print("  val", r, flush=True)

    cfg = saved_cfg or {"hidden": H, "d_lat": a.d_lat, "L": L, "scales": scales, "codebook": a.codebook, "P": d["cfg"]["P"],
           "quant": a.quant, "fsq_levels": levels, "pred": use_pred}
    save_compressor(comp.cpu(), cfg, a.out)
    comp.to(dev)
    comp.eval()
    with torch.no_grad():
        all_codes = [[] for _ in scales]
        for i in range(0, N, 128):
            _, codes, _ = comp.encode(hidden[i:i + 128].float().to(dev))
            for k, c in enumerate(codes):
                all_codes[k].append(c.cpu())
    d["codes"] = [torch.cat(c) for c in all_codes]
    d["val_idx"] = val
    torch.save(d, a.data)
    json.dump({"cfg": cfg, "history": hist, "seconds": round(time.time() - t0)}, open(a.out + ".json", "w"), indent=2)
    print(f"saved {a.out}; codes for {N} windows written into {a.data}")


if __name__ == "__main__":
    main()
