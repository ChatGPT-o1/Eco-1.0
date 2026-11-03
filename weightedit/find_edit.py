#!/usr/bin/env python3
"""Find a 'looping direction' in the MLP outputs and turn it into a weight edit (no gradients, no training).

Evidence: acts.pt from collect.py (per answer, per layer: mean MLP output over the answer's last tokens = 'tail', and over
the whole answer = 'full'). Within-answer drift d_i = tail_i - full_i cancels what an answer is ABOUT (topic, GSM8K vs
MATH); the looping direction per layer is mean drift of looping answers minus mean drift of the others. Layers are ranked
by how well that direction separates the two groups (gap / spread). In each chosen layer, the --cols columns of the MLP's
output matrix (down_proj: one column per neuron, the vector that neuron writes) that write most along the direction get
that direction removed: W[:, j] -= (W[:, j] . u) u. Same form of edit as the weight-space-programming post.

  python -I find_edit.py --acts practice/acts.pt --model Qwen/Qwen3.5-0.8B --layers 2 --cols 64 --out edits/mine.json
"""
import argparse, json, torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--loop-threshold", type=float, default=0.2, help="answers with loop_score above this count as looping")
    ap.add_argument("--layers", type=int, default=2, help="how many layers to edit (best separating)")
    ap.add_argument("--cols", type=int, default=64, help="neurons (down_proj columns) edited per layer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = torch.load(a.acts, weights_only=False)
    drift = (d["tail"].float() - d["full"].float())                 # (N, L, H)
    loop = d["loop_score"] > a.loop_threshold
    print(f"{int(loop.sum())} looping vs {int((~loop).sum())} other answers (loop_score > {a.loop_threshold})")
    assert loop.any() and (~loop).any(), "need answers in both groups"
    gap = drift[loop].mean(0) - drift[~loop].mean(0)                 # (L, H)
    u = gap / gap.norm(dim=-1, keepdim=True)
    proj = torch.einsum("nlh,lh->nl", drift, u)                       # each answer's drift along its layer's direction
    sep = (proj[loop].mean(0) - proj[~loop].mean(0)) / (proj.std(0) + 1e-6)
    order = sep.argsort(descending=True).tolist()
    print("layer separation (gap / spread):", " ".join(f"L{l}:{sep[l]:.2f}" for l in range(len(sep))))
    chosen = sorted(order[:a.layers])
    print("chosen layers:", chosen)

    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    spec = {"model": a.model, "rule": "W[:, j] -= (W[:, j] . u) u on down_proj columns", "loop_threshold": a.loop_threshold,
            "n_loop": int(loop.sum()), "n_other": int((~loop).sum()), "layers": {}}
    for l in chosen:
        W = lm.model.layers[l].mlp.down_proj.weight.detach()          # (H, neurons)
        c = u[l] @ W                                                   # how strongly each neuron writes along the direction
        cols = c.abs().argsort(descending=True)[:a.cols].tolist()
        frac = float((c[cols] ** 2).sum() / (c ** 2).sum())
        print(f"  layer {l}: separation {sep[l]:.2f}; {a.cols} neurons carry {frac:.0%} of what the layer writes along the direction")
        spec["layers"][str(l)] = {"direction": [round(x, 6) for x in u[l].tolist()], "columns": cols, "separation": round(float(sep[l]), 3)}
    spec["weights_changed"] = len(chosen) * a.cols * lm.config.hidden_size
    spec["of_total"] = sum(p.numel() for p in lm.parameters())
    json.dump(spec, open(a.out, "w"))
    print(f"edit: {spec['weights_changed']:,} weights ({100 * spec['weights_changed'] / spec['of_total']:.3f}% of {spec['of_total'] / 1e6:.0f}M) -> {a.out}")


if __name__ == "__main__":
    main()
