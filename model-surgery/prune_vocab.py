#!/usr/bin/env python3
"""Step 3: shrink the vocabulary to tokens your domain uses, and slice the embedding/LM-head rows.

Choose the cut either with a fixed count threshold (--min-count) or, preferably, with a token
inflation budget on held-out domain text (--max-inflation 0.015 means "held-out text may use at most
1.5% more tokens"); the largest threshold that fits the budget is picked from a sweep.

Lossless where it matters: text that only used kept tokens tokenizes to the same tokens, and the
model's logits on it are identical (up to renormalising over the smaller vocab). Only byte-level BPE
tokenizers are supported (Qwen, Llama-3, GPT-2/NeoX style); other tokenizers exit with a reason.

  python -I prune_vocab.py --src DIR --dst DIR --counts counts.npy \
      --heldout heldout.jsonl:text --max-inflation 0.015 [--keep-text must_keep.txt] [--dry-run]
"""
import argparse, glob, json, os, re, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vocabtools as vt

SWEEP = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 75, 90, 100, 125, 150, 200, 300, 500, 1000]
VOCAB_TENSOR = re.compile(r"(embed_tokens|lm_head|embed_out|wte|word_embeddings|^output|\.output)\b")
WEIGHT_FILES = re.compile(r"\.(safetensors|bin|pt|pth|gguf)$|\.safetensors\.index\.json$|^pytorch_model.*\.json$")


def load_heldout(spec, limit=3000):
    path, fields = spec.rsplit(":", 1) if ":" in os.path.basename(spec) else (spec, "text")
    fs = fields.split(",")
    out = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out.append("\n".join(str(r[f]) for f in fs if f in r))
        if len(out) >= limit:
            break
    return out


def vocab_rows(cfg):
    for d in (cfg, cfg.get("text_config") or {}):
        if d.get("vocab_size"):
            return d["vocab_size"]
    raise SystemExit("config has no vocab_size")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst")
    ap.add_argument("--counts", nargs="+", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--min-count", type=int)
    g.add_argument("--max-inflation", type=float, help="token-inflation budget on --heldout, e.g. 0.015")
    ap.add_argument("--heldout", action="append", default=[], help="PATH.jsonl[:FIELD,...] (default field: text)")
    ap.add_argument("--keep-text", help="file of text that must tokenize exactly (e.g. chat-template snippets, keywords)")
    ap.add_argument("--dry-run", action="store_true", help="print the sweep and exit")
    a = ap.parse_args()

    tj = vt.load_tokenizer_json(a.src)
    v = vt.Vocab(tj)
    old_tok = vt.tokenizer_from_json(tj)
    counts = sum(np.load(p) for p in a.counts)
    extra = []
    if a.keep_text:
        extra = sorted({i for e in old_tok.encode_batch(open(a.keep_text, encoding="utf-8").read().splitlines(),
                                                       add_special_tokens=False) for i in e.ids})
    held = [t for s in a.heldout for t in load_heldout(s)]
    if a.max_inflation is not None and not held:
        ap.error("--max-inflation needs --heldout")

    def trial(T):
        keep = v.keep_mask(counts, T, extra)
        new_tj, _, order = v.build(keep)
        infl, same = vt.inflation(old_tok, vt.tokenizer_from_json(new_tj), held)
        return keep, len(order), infl, same

    if a.min_count is not None:
        T = a.min_count
        keep, n_new, infl, same = trial(T)
        print(f"T={T}: vocab {n_new}  held-out inflation {infl:+.3%}  same-length docs {same:.1%}")
    else:
        print(f"{'T':>6} {'vocab':>8} {'inflation':>10} {'same-len':>9}")
        best = None
        for T in SWEEP:
            keep, n_new, infl, same = trial(T)
            print(f"{T:>6} {n_new:>8} {infl:>+10.3%} {same:>9.1%}", flush=True)
            if infl <= a.max_inflation:
                best = (T, keep, n_new, infl, same)
            else:
                break
        if best is None:
            raise SystemExit("even T=1 exceeds the inflation budget; raise --max-inflation or add counting data")
        T, keep, n_new, infl, same = best
        print(f"chosen T={T}: vocab {n_new} (was {v.n_ids}), inflation {infl:+.3%}")
    if a.dry_run:
        return
    if not a.dst:
        ap.error("--dst required unless --dry-run")

    new_tj, old2new, order = v.build(keep)
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    V_old = vocab_rows(cfg)
    V_new = len(order)
    warn = lambda m: print("WARNING:", m, file=sys.stderr)
    os.makedirs(a.dst, exist_ok=True)

    # weights: slice every tensor whose first dim is the vocab size and whose name looks like an embedding/head
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    files = sorted(glob.glob(os.path.join(a.src, "*.safetensors")))
    if not files:
        raise SystemExit("no .safetensors weights in --src (convert .bin checkpoints first)")
    idx = torch.as_tensor(order, dtype=torch.long)
    sliced = []
    total = 0
    for p in files:
        out = {}
        with safe_open(p, "pt") as f:
            meta = f.metadata() or {"format": "pt"}
            for k in f.keys():
                t = f.get_tensor(k)
                if t.shape and t.shape[0] == V_old and VOCAB_TENSOR.search(k) and t.dim() in (1, 2):
                    t = t.index_select(0, idx).contiguous()
                    sliced.append(k)
                out[k] = t
                total += t.numel() * t.element_size()
        save_file(out, os.path.join(a.dst, os.path.basename(p)), metadata=meta)
        del out
    if not sliced:
        raise SystemExit(f"found no embedding/lm_head tensor with {V_old} rows; nothing sliced")
    index = os.path.join(a.src, "model.safetensors.index.json")
    if os.path.exists(index):
        ij = json.load(open(index))
        ij.setdefault("metadata", {})["total_size"] = total
        json.dump(ij, open(os.path.join(a.dst, os.path.basename(index)), "w"), indent=2)

    # tokenizer + configs
    for f in os.listdir(a.src):
        s = os.path.join(a.src, f)
        if os.path.isfile(s) and not WEIGHT_FILES.search(f) and f not in ("tokenizer.json", "vocab.json", "merges.txt"):
            shutil.copy(s, a.dst)
    json.dump(new_tj, open(os.path.join(a.dst, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
    if os.path.exists(os.path.join(a.src, "vocab.json")):  # keep slow-tokenizer files consistent
        json.dump(new_tj["model"]["vocab"], open(os.path.join(a.dst, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    if os.path.exists(os.path.join(a.src, "merges.txt")):
        with open(os.path.join(a.dst, "merges.txt"), "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for m in new_tj["model"]["merges"]:
                f.write((m if isinstance(m, str) else " ".join(m)) + "\n")
    for name in ("config.json", "generation_config.json"):
        p = os.path.join(a.dst, name)
        if os.path.exists(p):
            json.dump(vt.remap_ids(json.load(open(p)), old2new, V_old, V_new, warn), open(p, "w"), indent=2)
    p = os.path.join(a.dst, "tokenizer_config.json")
    if os.path.exists(p):
        tc = json.load(open(p))
        if "added_tokens_decoder" in tc:
            tc["added_tokens_decoder"] = {str(int(old2new[int(k)])): val for k, val in tc["added_tokens_decoder"].items()
                                          if int(k) < len(old2new) and old2new[int(k)] >= 0}
        json.dump(tc, open(p, "w"), indent=2, ensure_ascii=False)
    json.dump({"min_count": T, "vocab_old": V_old, "vocab_new": V_new, "heldout_inflation": infl,
               "sliced_tensors": sliced, "kept_old_ids": order.tolist()},
              open(os.path.join(a.dst, "vocab_prune.json"), "w"))
    print(f"saved {a.dst}: vocab {V_old} -> {V_new}, sliced {sliced}")


if __name__ == "__main__":
    main()
