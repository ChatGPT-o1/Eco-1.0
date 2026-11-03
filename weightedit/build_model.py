#!/usr/bin/env python3
"""Build the edited Qwen3.5-0.8B by pure weight arithmetic (no training, no GPU).

    new = base + ALPHA * (style - base) + (loopfix - base)        # ALPHA defaults to 0: loop fix only

  base      Qwen/Qwen3.5-0.8B
  style     Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled: a LoRA fine-tune on 2.5k Claude Opus reasoning traces
            (merged weights). Its difference from the base is a "think like Opus" direction in weight space; adding it back
            to the base is the cheapest way to move the model toward that style without training anything ourselves.
  loopfix   RikiyaT/Qwen3.5-0.8B-weight-space-programming: the weight-space-programming edit that stops the thinking loops
            (64 down_proj columns in layers 15 and 19, 131,072 weights). Only the difference to the base is taken, so it
            sits on top of the style delta instead of replacing it.

The output has exactly the base model's tensor names, shapes and dtype (bfloat16), so it is a drop-in replacement.

  python -I build_model.py --out final/                  # the loop fix only (default)
  python -I build_model.py --out final/ --alpha 0.3      # loop fix plus 30% of the Opus-style delta
  python -I build_model.py --out final/ --alpha 1.0      # loop fix plus the Opus-style fine-tune at full strength
"""
import argparse, glob, json, os, shutil, sys

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

# pinned so the same recipe always gives the same weights
BASE = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
STYLE = ("Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled", "af2f37de41af0bdcaea5e3790ad323030ea4af07")
LOOPFIX = ("RikiyaT/Qwen3.5-0.8B-weight-space-programming", "d1671a178705bcab7644806d23c535517f35723b")


def open_all(repo, rev):
    path = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
    files = [safe_open(f, "pt") for f in sorted(glob.glob(os.path.join(path, "*.safetensors")))]
    return {k: f for f in files for k in f.keys()}, path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.0, help="how much of the Opus-style delta to add (0 = none, 1.0 = the fine-tune itself)")
    ap.add_argument("--no-loopfix", action="store_true", help="skip the loop-removal edit")
    a = ap.parse_args()

    base, base_path = open_all(*BASE)
    style = open_all(*STYLE)[0] if a.alpha else None
    loop = open_all(*LOOPFIX)[0] if not a.no_loopfix else None
    for name, other in (("style", style), ("loopfix", loop)):
        if other is not None and set(other) != set(base):
            sys.exit(f"{name} model has different tensor names than the base: refusing to merge")

    out, report = {}, {"alpha": a.alpha, "style": STYLE if a.alpha else None, "loopfix": None if a.no_loopfix else LOOPFIX,
                       "tensors_changed": 0, "style_tensors": 0, "loop_tensors": {}, "style_rel_change": 0.0}
    sq_delta = sq_base = 0.0
    for k in sorted(base):
        b = base[k].get_tensor(k)
        w = b.float()
        if style is not None:
            d = style[k].get_tensor(k).float() - w
            if d.abs().max() > 0:
                w = w + a.alpha * d
                report["style_tensors"] += 1
                sq_delta += float((a.alpha * d).norm() ** 2)
                sq_base += float(b.float().norm() ** 2)
        if loop is not None:
            d = loop[k].get_tensor(k).float() - b.float()
            if d.abs().max() > 0:
                w = w + d
                report["loop_tensors"][k] = int((d.abs().sum(0) > 0).sum())
        out[k] = w.to(b.dtype).contiguous()
        report["tensors_changed"] += int(not torch.equal(out[k], b))
    report["style_rel_change"] = round((sq_delta / max(sq_base, 1e-12)) ** 0.5, 5)

    bad = [k for k, v in out.items() if not torch.isfinite(v.float()).all()]
    if bad:
        sys.exit(f"non-finite values in {bad[:3]}")

    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(base_path):  # config, tokenizer, chat template: copied from the base; weights are rewritten below
        if not f.endswith(".safetensors") and not f.endswith(".index.json") and os.path.isfile(os.path.join(base_path, f)):
            shutil.copy(os.path.join(base_path, f), os.path.join(a.out, f))
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    json.dump(report, open(os.path.join(a.out, "edit_recipe.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
