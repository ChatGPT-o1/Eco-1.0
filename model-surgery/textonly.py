#!/usr/bin/env python3
"""Step 1: save a text-only copy of a model (drops vision towers and unused MTP/NextN heads).

Loads with AutoModelForCausalLM, which for multimodal checkpoints (e.g. Qwen3.5) builds only the
language model, then saves it in the source dtype. If the saved weights have no MTP/NextN tensors,
the config's MTP layer count is set to 0 so llama.cpp doesn't expect a missing block.

  python -I textonly.py SRC_DIR_OR_HF_ID DST_DIR
"""
import argparse, glob, json, os, shutil, sys, time


MTP_KEYS = ("mtp_num_hidden_layers", "num_nextn_predict_layers")


def zero_mtp(cfg):
    changed = []
    for d in [cfg, cfg.get("text_config") or {}]:
        for k in MTP_KEYS:
            if d.get(k):
                d[k] = 0
                changed.append(k)
    return changed


def saved_tensor_names(dst):
    from safetensors import safe_open
    names = []
    for p in glob.glob(os.path.join(dst, "*.safetensors")):
        with safe_open(p, "pt") as f:
            names += list(f.keys())
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    t = time.time()
    src_cfg = AutoConfig.from_pretrained(a.src)
    model = AutoModelForCausalLM.from_pretrained(a.src, dtype="auto")
    tok = AutoTokenizer.from_pretrained(a.src)
    n_params = sum(p.numel() for p in model.parameters())
    os.makedirs(a.dst, exist_ok=True)
    model.save_pretrained(a.dst)
    tok.save_pretrained(a.dst)
    # carry over chat template / generation config files that save_pretrained may not write
    if os.path.isdir(a.src):
        for f in ("chat_template.jinja", "chat_template.json", "generation_config.json"):
            s = os.path.join(a.src, f)
            if os.path.exists(s) and not os.path.exists(os.path.join(a.dst, f)):
                shutil.copy(s, a.dst)

    names = saved_tensor_names(a.dst)
    has_mtp = any(".mtp." in n or n.startswith("mtp.") or "nextn" in n for n in names)
    cfg_path = os.path.join(a.dst, "config.json")
    cfg = json.load(open(cfg_path))
    changed = [] if has_mtp else zero_mtp(cfg)
    if changed:
        json.dump(cfg, open(cfg_path, "w"), indent=2)

    print(json.dumps({
        "src_architectures": getattr(src_cfg, "architectures", None),
        "saved_class": type(model).__name__,
        "params_M": round(n_params / 1e6, 1),
        "has_mtp_tensors": has_mtp,
        "config_keys_zeroed": changed,
        "seconds": round(time.time() - t, 1),
        "dst": a.dst,
    }, indent=2))


if __name__ == "__main__":
    main()
