"""Loading/saving for llm2ncm artifacts (work dir layout):
  data.pt          windows: prefix/block token ids, encoder hidden states, codes (after compression)
  compressor.pt    Compressor weights + config
  generator/       LoRA adapter (llm backbone) + extras.pt (new modules) + config.json
"""
import json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncm_model import Compressor, NCMGenerator  # noqa: E402

END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>", "<|end|>", "</s>")


def load_llm(model_dir, threads=None, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if threads:
        torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_dir)
    lm = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
    return tok, lm


def pick_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def end_ids(tok):
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for t in END_TOKENS:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    return ids


def save_compressor(comp, cfg, path):
    torch.save({"state": comp.state_dict(), "cfg": cfg}, path)


def load_compressor(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    c = d["cfg"]
    comp = Compressor(c["hidden"], c["d_lat"], c["L"], c["scales"], c["codebook"])
    comp.load_state_dict(d["state"])
    return comp.eval(), c


def save_generator(gen, cfg, path):
    os.makedirs(path, exist_ok=True)
    if cfg["backbone"] == "llm":
        gen.lm.save_pretrained(os.path.join(path, "lora"))
    torch.save(gen.extra_state(), os.path.join(path, "extras.pt"))
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"), indent=2)


def load_generator(model_dir, path, threads=None, device="cpu"):
    """Returns (tokenizer, generator, compressor, cfg). The generator's .lm with the adapter disabled is
    the original LLM, which ncm_eval.py uses as the baseline/judge."""
    cfg = json.load(open(os.path.join(path, "config.json")))
    tok, lm = load_llm(model_dir, threads, device)
    if cfg["backbone"] == "llm":
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, os.path.join(path, "lora")).to(device)
    comp, ccfg = load_compressor(cfg["compressor"])
    cfg["compressor_cfg"] = ccfg
    gen = NCMGenerator(lm, ccfg["d_lat"], ccfg["scales"], ccfg["L"], ccfg["codebook"], backbone=cfg["backbone"])
    missing = gen.load_state_dict(torch.load(os.path.join(path, "extras.pt"), map_location="cpu"), strict=False)
    bad = [k for k in missing.missing_keys if not k.startswith("lm.")]
    if bad:
        raise SystemExit(f"generator extras missing keys: {bad[:5]}")
    for p in gen.parameters():
        p.requires_grad_(False)
    return tok, gen.to(device).eval(), comp.to(device), cfg
