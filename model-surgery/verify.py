#!/usr/bin/env python3
"""Step 4: check a vocab-pruned model against its source.

  1. Tokenization: share of held-out docs that tokenize to exactly the same tokens, and total inflation.
  2. Logits: on held-out texts that use no dropped tokens, the pruned model's log-probs (renormalised
     over the kept vocab) must match the source model's. Expect ~0 difference.
  3. Generation: one greedy chat reply from the pruned model, to catch broken templates/eos ids.
Exits non-zero if logits disagree beyond --tol.

  python -I verify.py --orig SRC_DIR --pruned DST_DIR --heldout heldout.jsonl:text
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from prune_vocab import load_heldout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--pruned", required=True)
    ap.add_argument("--heldout", action="append", required=True)
    ap.add_argument("--n-logit-texts", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--prompt", default="Write a Python function that reverses a string.")
    ap.add_argument("--out")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(a.threads)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    info = json.load(open(os.path.join(a.pruned, "vocab_prune.json")))
    kept = np.array(info["kept_old_ids"], dtype=np.int64)
    old2new = -np.ones(max(int(kept.max()) + 1, info["vocab_old"]), dtype=np.int64)
    old2new[kept] = np.arange(len(kept))
    t_old = AutoTokenizer.from_pretrained(a.orig)
    t_new = AutoTokenizer.from_pretrained(a.pruned)
    texts = [t for s in a.heldout for t in load_heldout(s)]
    report = {"pruned": a.pruned, "vocab_old": info["vocab_old"], "vocab_new": info["vocab_new"]}

    exact = longer = 0
    so = sn = 0
    clean = []
    for t in texts:
        o = t_old(t, add_special_tokens=False)["input_ids"]
        n = t_new(t, add_special_tokens=False)["input_ids"]
        so += len(o); sn += len(n)
        mapped = old2new[o] if len(o) else np.array([], dtype=np.int64)
        if (mapped >= 0).all() and list(mapped) == n:
            exact += 1
            if len(o) >= 32 and len(clean) < a.n_logit_texts:
                clean.append(o[: a.max_tokens])
        elif (mapped >= 0).all():
            print("ERROR: doc uses only kept tokens but tokenizes differently", file=sys.stderr)
            report["tokenization_bug"] = True
        longer += len(n) > len(o)
    report["tokenization"] = {"docs": len(texts), "exact": exact / max(len(texts), 1),
                              "inflation": sn / max(so, 1) - 1, "docs_longer": longer}
    print("tokenization:", report["tokenization"])

    def logprobs(model_dir, seqs, rows=None):
        m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).eval()
        out = []
        with torch.inference_mode():
            for s in seqs:
                lg = m(input_ids=torch.tensor([s])).logits[0].float()
                if rows is not None:
                    lg = lg[:, rows]
                out.append(torch.log_softmax(lg, -1))
        del m
        return out

    if clean:
        ref = logprobs(a.orig, clean, rows=torch.as_tensor(kept))
        new = logprobs(a.pruned, [list(old2new[s]) for s in clean])
        diff = max(float((x - y).abs().max()) for x, y in zip(ref, new))
        agree = float(np.mean([float((x.argmax(-1) == y.argmax(-1)).float().mean()) for x, y in zip(ref, new)]))
        report["logits"] = {"texts": len(clean), "max_abs_logprob_diff": diff, "argmax_agreement": agree}
        print("logits:", report["logits"])
    else:
        print("WARNING: no held-out text without dropped tokens; logits check skipped", file=sys.stderr)

    m = AutoModelForCausalLM.from_pretrained(a.pruned, dtype=dtype).eval()
    enc = t_new.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True,
                                    enable_thinking=False, return_tensors="pt", return_dict=True)
    with torch.inference_mode():
        out = m.generate(**enc, max_new_tokens=96, do_sample=False, repetition_penalty=1.05)
    reply = t_new.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    report["generation"] = {"prompt": a.prompt, "reply": reply}
    print("generation:", reply[:400])

    ok = not report.get("tokenization_bug") and report.get("logits", {}).get("max_abs_logprob_diff", 0) <= a.tol
    report["ok"] = ok
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("VERIFY", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
