#!/usr/bin/env python3
"""Generate thinking answers with Qwen3.5-0.8B, score them, and (optionally) record what each MLP writes.

For every problem: one sampled thinking answer (temperature 0.6, top-p 0.95, top-k 20, as in the weight-space
programming post) capped at --max-new tokens; whether it hit the cap (our proxy for "looped"), an n-gram repetition
score, the \\boxed{} answer and whether it is correct. With --acts, a second forward pass over each full answer records
the mean MLP output of every layer over the answer's LAST --act-tail tokens (where loops live) and over all of it:
that is the evidence used to find a "looping direction" without any gradients.

  python -I collect.py --model DIR --split practice --n 256 --out practice/ --acts --device cuda
  python -I collect.py --model DIR --split math500 --n 100 --out eval_base/ --device cuda
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import chat_prompt, is_correct, last_boxed, load_problems, loop_score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True, choices=["practice", "math500"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--acts", action="store_true", help="also record per-layer mean MLP outputs for each answer")
    ap.add_argument("--act-tail", type=int, default=1024)
    ap.add_argument("--act-batch", type=int, default=4)
    ap.add_argument("--time-budget-min", type=float, default=0.0, help="stop starting new generation batches after this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    dt = torch.float16 if a.device == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(a.device).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    probs = load_problems(a.split, a.n, a.seed, cache=os.path.join(a.out, "_hf"))
    print(f"{len(probs)} {a.split} problems; generating up to {a.max_new} tokens each", flush=True)

    t0 = time.time()
    rows = []
    with open(os.path.join(a.out, "answers.jsonl"), "w") as f:
        for i in range(0, len(probs), a.batch):
            if a.time_budget_min and time.time() - t0 > a.time_budget_min * 60:
                print("time budget reached: stopping generation", flush=True)
                break
            group = probs[i:i + a.batch]
            enc = tok([chat_prompt(tok, p["problem"]) for p in group], return_tensors="pt", padding=True).to(a.device)
            with torch.inference_mode():
                out = lm.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                  pad_token_id=pad)
            P = enc["input_ids"].shape[1]
            for p, seq, att in zip(group, out, enc["attention_mask"]):
                prompt_ids = seq[:P][att.bool()].tolist()
                gen = seq[P:].tolist()
                eos_at = next((j for j, t in enumerate(gen) if t in (tok.eos_token_id, pad) or t == tok.convert_tokens_to_ids("<|im_end|>")), None)
                hit_cap = eos_at is None
                gen = gen if hit_cap else gen[:eos_at + 1]
                text = tok.decode(gen, skip_special_tokens=False)
                after = text.split("</think>")[-1] if "</think>" in text else ""
                pred = last_boxed(after) or last_boxed(text)
                r = {"id": p["id"], "source": p["source"], "gold": p["answer"], "pred": pred, "correct": is_correct(pred, p["answer"]),
                     "tokens": len(gen), "hit_cap": hit_cap, "closed_think": "</think>" in text, "loop_score": round(loop_score(gen), 4),
                     "prompt_ids": prompt_ids, "gen_ids": gen}
                rows.append(r)
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("prompt_ids", "gen_ids")}) + "\n")
            done = len(rows)
            print(f"  {done}/{len(probs)} answered ({time.time() - t0:.0f}s): hit cap {sum(r['hit_cap'] for r in rows) / done:.0%}, "
                  f"correct {sum(r['correct'] for r in rows) / done:.0%}", flush=True)

    n = len(rows)
    summary = {"split": a.split, "n": n, "max_new": a.max_new, "hit_cap": round(sum(r["hit_cap"] for r in rows) / n, 4),
               "correct": round(sum(r["correct"] for r in rows) / n, 4), "mean_tokens": round(sum(r["tokens"] for r in rows) / n, 1),
               "mean_loop_score": round(sum(r["loop_score"] for r in rows) / n, 4), "seconds": round(time.time() - t0)}
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        summary[src] = {"n": len(sub), "correct": round(sum(r["correct"] for r in sub) / len(sub), 4),
                        "hit_cap": round(sum(r["hit_cap"] for r in sub) / len(sub), 4)}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)

    if a.acts:  # mean MLP output per layer: over the answer's last act_tail tokens, and over the whole answer
        layers = lm.model.layers
        L, H = len(layers), lm.config.hidden_size
        tail = torch.zeros(n, L, H, dtype=torch.float16)
        full = torch.zeros(n, L, H, dtype=torch.float16)
        cur = {}
        hooks = [layers[l].mlp.register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(l, o)) for l in range(L)]
        tok.padding_side = "right"
        t1 = time.time()
        with torch.inference_mode():
            for i in range(0, n, a.act_batch):
                grp = rows[i:i + a.act_batch]
                seqs = [r["prompt_ids"] + r["gen_ids"] for r in grp]
                T = max(len(s) for s in seqs)
                ids = torch.full((len(seqs), T), pad, dtype=torch.long)
                att = torch.zeros(len(seqs), T, dtype=torch.long)
                for b, s in enumerate(seqs):
                    ids[b, :len(s)] = torch.tensor(s)
                    att[b, :len(s)] = 1
                lm.model(input_ids=ids.to(a.device), attention_mask=att.to(a.device))
                for b, r in enumerate(grp):
                    s, e = len(r["prompt_ids"]), len(r["prompt_ids"]) + len(r["gen_ids"])
                    for l in range(L):
                        o = cur[l][b].float()
                        full[i + b, l] = o[s:e].mean(0).half().cpu()
                        tail[i + b, l] = o[max(s, e - a.act_tail):e].mean(0).half().cpu()
                if (i // a.act_batch) % 16 == 0:
                    print(f"  activations {min(i + a.act_batch, n)}/{n} ({time.time() - t1:.0f}s)", flush=True)
        for h in hooks:
            h.remove()
        torch.save({"tail": tail, "full": full, "hit_cap": torch.tensor([r["hit_cap"] for r in rows]),
                    "correct": torch.tensor([r["correct"] for r in rows]), "loop_score": torch.tensor([r["loop_score"] for r in rows]),
                    "ids": [r["id"] for r in rows], "act_tail": a.act_tail}, os.path.join(a.out, "acts.pt"))
        print(f"saved activations {tuple(tail.shape)} for {n} answers", flush=True)


if __name__ == "__main__":
    main()
