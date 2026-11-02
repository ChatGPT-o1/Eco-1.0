#!/usr/bin/env python3
"""Benchmark the draft head: real wall-clock speed versus plain greedy decoding, and proof of exactness.

For every prompt it runs plain greedy decoding (one forward pass per token), speculative decoding with the head at each
--k, and Hugging Face generate() as an outside reference; it checks token-for-token that speculative == plain. Prompts:
held-out chat prompts (fetch_prompts.py, never trained on), the three questions from this project, and HumanEval
problems if humaneval.parquet is present.

  python -I bench_draft.py --model DIR --head head.pt --data msdata/ --device cuda --out bench.json
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from eagle import EagleHead, end_ids, greedy_plain, greedy_spec

QUESTIONS = ["What is 2 + 2?", "What is the capital of France?", "Write a Python function that adds two numbers."]


def load_prompts(tok, data, n):
    msgs = [[{"role": "user", "content": q}] for q in QUESTIONS]
    hp = os.path.join(data, "heldout_prompts.jsonl")
    if os.path.exists(hp):
        rows = [json.loads(l) for l in open(hp) if l.strip()]
        half = len(rows) // 2  # first half smol-smoltalk, second half alpaca: alternate
        mixed = [r for pair in zip(rows[:half], rows[half:]) for r in pair]
        msgs += [r["messages"] for r in mixed[:n]]
    he = os.path.join(data, "humaneval.parquet")
    if os.path.exists(he):
        import pyarrow.parquet as pq
        for row in pq.read_table(he).to_pylist()[:max(4, n // 3)]:
            msgs.append([{"role": "user", "content": "Complete this Python function:\n```python\n" + row["prompt"] + "```"}])
    return [tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"] for m in msgs]


def timed(fn, dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return r, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=60, help="held-out chat prompts (plus the 3 questions and some HumanEval)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--k", default="2,4,6", help="draft lengths to test")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = torch.float16 if a.dtype == "float16" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(a.device).eval()
    ck = torch.load(a.head, map_location="cpu", weights_only=False)
    head = EagleHead(lm.config)
    head.load_state_dict(ck["state"])
    head = head.to(a.device, dt).eval()
    eos = end_ids(tok)
    prompts = load_prompts(tok, a.data, a.n)
    ks = [int(x) for x in a.k.split(",")]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    for p in prompts[:2]:  # warm-up (GPU kernels, allocator)
        greedy_plain(lm, p, 32, eos)
        greedy_spec(lm, head, p, 32, eos, k=ks[0])

    plain_t = plain_tok = hf_t = 0.0
    res = {k: {"time": 0.0, "tokens": 0, "passes": 0, "rounds": 0, "accepted": 0, "exact": 0} for k in ks}
    answers = []
    for i, p in enumerate(prompts):
        (ref, _), t_plain = timed(lambda: greedy_plain(lm, p, a.max_new, eos), a.device)
        plain_t += t_plain
        plain_tok += len(ref)
        _, t_hf = timed(lambda: lm.generate(torch.tensor([p], device=a.device), attention_mask=torch.ones(1, len(p), dtype=torch.long, device=a.device),
                                             max_new_tokens=a.max_new, do_sample=False, eos_token_id=sorted(eos), pad_token_id=pad), a.device)
        hf_t += t_hf
        for k in ks:
            (got, passes, acc), t_spec = timed(lambda: greedy_spec(lm, head, p, a.max_new, eos, k=k), a.device)
            r = res[k]
            r["time"] += t_spec
            r["tokens"] += len(got)
            r["passes"] += passes
            r["rounds"] += len(acc)
            r["accepted"] += sum(acc)
            r["exact"] += int(got == ref)
        if i < len(QUESTIONS):
            answers.append({"question": QUESTIONS[i], "answer": tok.decode(ref, skip_special_tokens=True)})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts", flush=True)

    plain_tps = plain_tok / plain_t
    out = {"dtype": a.dtype, "device": a.device, "prompts": len(prompts), "max_new": a.max_new, "head_step": ck.get("step"),
           "head_val": ck.get("val"), "plain_tokens_per_s": round(plain_tps, 1), "hf_generate_tokens_per_s": round(plain_tok / hf_t, 1),
           "per_k": {}, "answers": answers}
    rows = [f"| decoder | exact match | accepted drafts / round | tokens per pass | tokens/s | speed-up vs plain | vs HF generate |",
            "|---|---|---|---|---|---|---|",
            f"| plain greedy (1 token per pass) | - | - | 1.00 | {plain_tps:.1f} | 1.00x | {plain_tps / (plain_tok / hf_t):.2f}x |"]
    for k in ks:
        r = res[k]
        tps = r["tokens"] / r["time"]
        out["per_k"][k] = {"exact_match": f"{r['exact']}/{len(prompts)}", "accepted_per_round": round(r["accepted"] / max(r["rounds"], 1), 3),
                           "tokens_per_pass": round(r["tokens"] / max(r["passes"], 1), 3), "tokens_per_s": round(tps, 1),
                           "speedup_vs_plain": round(tps / plain_tps, 3), "speedup_vs_hf": round(tps / (plain_tok / hf_t), 3)}
        o = out["per_k"][k]
        rows.append(f"| draft head, k={k} | {o['exact_match']} | {o['accepted_per_round']:.2f} of {k} | {o['tokens_per_pass']:.2f} | "
                    f"{tps:.1f} | **{o['speedup_vs_plain']:.2f}x** | {o['speedup_vs_hf']:.2f}x |")
    out["table"] = "\n".join(rows)
    print(out["table"])
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
