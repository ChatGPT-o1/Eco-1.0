#!/usr/bin/env python3
"""Take a random subset of the windows in a build_data.py output (to compare training-set sizes from one build).

build_data.py lists all code windows first and the chat windows after them, so a plain "first n" slice would
drop the chat windows entirely; this samples with a fixed seed instead and keeps the code/chat/distilled mix.
Run it on data.pt BEFORE train_compressor.py (which adds codes to the file).

  python -I subset_data.py --data big.pt --out small.pt --n 2000
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = torch.load(a.data, weights_only=False)
    assert "codes" not in d, "subset before train_compressor.py, which adds codes/val_idx"
    N = len(d["block"])
    keep = torch.randperm(N, generator=torch.Generator().manual_seed(a.seed))[:min(a.n, N)].sort().values
    out = dict(d)
    for k in ("prefix", "block", "hidden", "distilled"):
        out[k] = d[k][keep].clone()  # clone: a view would make torch.save write the whole original storage
    out["source"] = [d["source"][i] for i in keep.tolist()]
    torch.save(out, a.out)
    n_chat = sum(s == "chat" for s in out["source"])
    print(f"subset {len(keep)}/{N} windows ({n_chat} chat, {int(out['distilled'].sum())} distilled) -> {a.out}")


if __name__ == "__main__":
    main()
