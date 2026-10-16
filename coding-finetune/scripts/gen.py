"""Batched generation for Qwen3.5-2B (+ optional LoRA adapter) on CPU."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from schema import BASE_MODEL, render

torch.set_num_threads(4)


def load(adapter=None):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tok.convert_tokens_to_ids("<|im_end|>"))
    model._eos = sorted(eos)
    return model, tok


def generate(model, tok, items, max_new=640, batch=8, temperature=0.0, top_p=0.95, seed=0, log=None):
    """items: list of dicts with 'messages' and optional 'tools'. Returns list of (text, finished)."""
    prompts = [render(tok, it["messages"], it.get("tools"), gen=True) for it in items]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    t0 = time.time()
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                  eos_token_id=model._eos)
        if temperature > 0:
            torch.manual_seed(seed + s)
            kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            res = model.generate(**enc, **kw)
        L = enc["input_ids"].shape[1]
        for i, row in zip(idx, res):
            g = row[L:].tolist()
            fin = any(t in model._eos for t in g)
            out[i] = (tok.decode(g, skip_special_tokens=True).strip(), fin)
        if log:
            log(f"  gen {min(s + batch, len(order))}/{len(order)} {time.time() - t0:.0f}s")
    return out
