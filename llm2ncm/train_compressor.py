#!/usr/bin/env python3
"""Stage A: train the multiscale residual-VQ compressor on the frozen LLM's hidden states.

Loss = token cross-entropy of a small decoder (through the LLM's frozen output layer) reconstructing the
block from the code levels + VQ commitment. Nested dropout trains with only the first k levels half the
time, so coarse levels carry the most information. Then every window is encoded to codes for stage B.

  python -I train_compressor.py --model DIR --data work/data.pt --out work/compressor.pt --scales 2,8
"""
import argparse, json, os, random, sys, time

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
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=0)
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
    perm = torch.randperm(N)
    nval = max(16, int(N * a.val_frac))
    val, tr = perm[:nval], perm[nval:]
    hidden, block = d["hidden"], d["block"].long()
    comp = Compressor(H, a.d_lat, L, scales, a.codebook).to(dev)
    opt = torch.optim.AdamW(comp.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))

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
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(comp.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"step {step} ce {ce.item():.3f} commit {commit.item():.4f} levels {nl} ({time.time() - t0:.0f}s)", flush=True)
        if step % 500 == 0 or step == a.steps:
            r = evaluate()
            r["step"] = step
            hist.append(r)
            print("  val", r, flush=True)

    cfg = {"hidden": H, "d_lat": a.d_lat, "L": L, "scales": scales, "codebook": a.codebook, "P": d["cfg"]["P"]}
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
