#!/usr/bin/env python3
"""Training data for the draft head: the ORIGINAL model's own greedy answers to chat prompts.

The draft head must guess what this model will say next, so its training text should be this model's own output
(EAGLE does the same with the target model's responses). Prompts come from prompts.jsonl (llm2ncm/fetch_prompts.py:
smol-smoltalk + alpaca) plus, if present, coding instructions from count_chat.jsonl (model-surgery/fetch_data.py).

  python -I build_draft_data.py --model DIR --data msdata/ --out draft_data.pt --n 12000 --device cuda
"""
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from eagle import end_ids


def read_jsonl(path, limit):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dir with prompts.jsonl (and optionally count_chat.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12000, help="number of prompts to answer")
    ap.add_argument("--code-share", type=float, default=0.3, help="share of coding instructions (if count_chat.jsonl exists)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-prompt", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16 if a.device == "cuda" else torch.float32).to(a.device).eval()
    stops = sorted(end_ids(tok))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    msgs = [r["messages"] for r in read_jsonl(os.path.join(a.data, "prompts.jsonl"), a.n * 2)]
    cc = os.path.join(a.data, "count_chat.jsonl")
    if os.path.exists(cc):
        rows = read_jsonl(cc, a.n)
        rng.shuffle(rows)
        k = int(a.n * a.code_share)
        msgs = msgs[:a.n - k] + [[{"role": "user", "content": r["instruction"]}] for r in rows[:k]]
    rng.shuffle(msgs)
    prompts = []
    for m in msgs:
        ids = tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        if len(ids) <= a.max_prompt:
            prompts.append(ids)
        if len(prompts) >= a.n:
            break
    prompts.sort(key=len)  # similar lengths per batch -> little padding (order is shuffled again below)

    t0 = time.time()
    seqs, resp_from = [], []
    with torch.inference_mode():
        for i in range(0, len(prompts), a.batch):
            group = prompts[i:i + a.batch]
            n_max = max(len(p) for p in group)
            ids = torch.tensor([[pad] * (n_max - len(p)) + p for p in group], device=a.device)
            att = torch.tensor([[0] * (n_max - len(p)) + [1] * len(p) for p in group], device=a.device)
            out = lm.generate(ids, attention_mask=att, max_new_tokens=a.max_new, do_sample=False, eos_token_id=stops, pad_token_id=pad)
            for p, r in zip(group, out[:, n_max:].tolist()):
                end = next((j for j, t in enumerate(r) if t in stops), None)
                r = r[:end + 1] if end is not None else r
                if len(r) >= 4:
                    seqs.append(p + r)
                    resp_from.append(len(p))
            if (i // a.batch) % 10 == 0:
                print(f"  answered {min(i + a.batch, len(prompts))}/{len(prompts)} prompts ({time.time() - t0:.0f}s)", flush=True)
    order = list(range(len(seqs)))
    rng.shuffle(order)
    seqs, resp_from = [seqs[j] for j in order], [resp_from[j] for j in order]
    n_resp = sum(len(s) - r for s, r in zip(seqs, resp_from))
    torch.save({"seqs": seqs, "resp_from": resp_from, "cfg": {"model": a.model, "max_new": a.max_new}}, a.out)
    print(f"saved {a.out}: {len(seqs)} answers, {n_resp} response tokens, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
