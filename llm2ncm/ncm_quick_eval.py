#!/usr/bin/env python3
"""Fast, low-noise end-to-end check of a converted NCM, for comparing training changes.

ncm_eval.py scores 16 prompts, far too few to see whether a training tweak helped. This scores one block on
each of N held-out windows from data.pt (the compressor's validation split, which neither the compressor nor
the generator trained on). For every window the NCM writes a block from the prefix alone, and we report,
separately for windows whose block is the original LLM's own greedy continuation ("distilled") and for real text:

  agree / match   share of tokens equal to the block in data.pt (the teacher's greedy output for distilled
                  windows, the real continuation otherwise)
  judge_bits      the ORIGINAL LLM's bits/token on the NCM's block given the prefix (lower is better);
                  ref_bits is the same score for the block in data.pt: the floor for distilled, normal text for real
  oracle_*        same, but the generator is handed the TRUE codes instead of its own predictions. The gap between
                  oracle_agree and agree is the cost of predicting codes; a low oracle_agree means the codes plus
                  mask-predict cannot spell the tokens out (compressor too lossy or generator too weak)

  python -I ncm_quick_eval.py --model DIR --generator work/generator --data work/data.pt --out work/quick_eval.json
"""
import argparse, contextlib, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from ncm_io import load_generator, pick_device

LN2 = 0.6931471805599453


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--data", required=True, help="work/data.pt after train_compressor.py (needs codes and val_idx)")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=256, help="held-out windows")
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    a = ap.parse_args()
    dev = pick_device(a.device)
    tok, gen, comp, cfg = load_generator(a.model, a.generator, a.threads, dev)
    lm = gen.lm
    teacher = lm.disable_adapter if hasattr(lm, "disable_adapter") else contextlib.nullcontext
    d = torch.load(a.data, weights_only=False)
    assert "codes" in d and "val_idx" in d, "run train_compressor.py first"
    L = d["cfg"]["L"]
    idx = d["val_idx"][:a.n].tolist()

    def judge_bits(prefix, cont):
        with teacher(), torch.inference_mode():
            ids = torch.tensor([prefix + cont], device=dev)
            logits = lm(input_ids=ids).logits[0, len(prefix) - 1:-1].float()
            return float(F.cross_entropy(logits, torch.tensor(cont, device=dev), reduction="sum")) / LN2

    acc = {g: {"n": 0, "match": 0.0, "oracle_match": 0.0, "bits": 0.0, "ref_bits": 0.0, "toks": 0} for g in ("distilled", "real")}
    t0 = time.time()
    passes = 0
    outs, refs, samples = [], [], []
    for j, i in enumerate(idx):
        prefix = d["prefix"][i].long()
        if "prefix_mask" in d:
            prefix = prefix[d["prefix_mask"][i]]  # drop left padding: generation sees the real prompt only
        prefix = prefix.tolist()
        ref = d["block"][i].long().tolist()
        g = acc["distilled" if bool(d["distilled"][i]) else "real"]
        out, p = gen.generate(comp.rvq, prefix, L, refine=a.refine)
        oracle = [d["codes"][k][i:i + 1].long().to(dev) for k in range(len(d["codes"]))]
        out_o, _ = gen.generate(comp.rvq, prefix, L, refine=a.refine, oracle_codes=oracle)
        out, out_o = (out + ref)[:L], (out_o + ref)[:L]  # pad (never needed with end_ids=()) so zip is length L
        passes += p
        outs.append(out)
        refs.append(ref)
        if len(samples) < 6:
            samples.append({"group": "distilled" if bool(d["distilled"][i]) else "real", "prefix_end": tok.decode(prefix[-24:]),
                            "ncm": tok.decode(out), "reference": tok.decode(ref)})
        g["n"] += 1
        g["match"] += sum(x == y for x, y in zip(out, ref)) / L
        g["oracle_match"] += sum(x == y for x, y in zip(out_o, ref)) / L
        g["bits"] += judge_bits(prefix, out)
        g["ref_bits"] += judge_bits(prefix, ref)
        g["toks"] += L
        if (j + 1) % 64 == 0:
            print(f"  quick_eval {j + 1}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)

    def distinct2(blocks):  # share of distinct token bigrams; collapses toward 0 when the output loops ("add add add ...")
        grams = [tuple(b[k:k + 2]) for b in blocks for k in range(len(b) - 1)]
        return round(len(set(grams)) / max(len(grams), 1), 3)

    res = {"n": len(idx), "refine": a.refine, "passes_per_block": round(passes / max(len(idx), 1), 2),
           "distinct2": distinct2(outs), "ref_distinct2": distinct2(refs), "samples": samples}
    for name, g in acc.items():
        if g["n"]:
            res[name] = {"n": g["n"], "agree": round(g["match"] / g["n"], 4), "oracle_agree": round(g["oracle_match"] / g["n"], 4),
                         "judge_bits": round(g["bits"] / g["toks"], 3), "ref_bits": round(g["ref_bits"] / g["toks"], 3)}
    print("QUICK_EVAL " + json.dumps({k: v for k, v in res.items() if k != "samples"}), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
