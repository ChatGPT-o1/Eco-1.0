#!/usr/bin/env python3
"""Two one-off tables for the training pool (MBPP train, 374 problems), written to W/:

  screen.json   pass counts of SmolLM2 itself: --k samples per problem at T=1 (the sampler's starting prior)
  teacher.json  for every problem, up to 2 test-passing solutions from Qwen2.5-Coder-0.5B-Instruct (4 samples at T=0.7),
                re-tokenised for SmolLM2 as a ```python block. Used by V8 only, and only for problems the student keeps failing.
Reference solutions are never used as answers.

  python -I bank.py screen  --out W
  python -I bank.py teacher --out W
"""
import argparse, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

TEACHER = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")


def screen(a):
    tok, model = E.load(dtype=torch.bfloat16)
    probs = E.problems("train")
    prompts = [E.chat_ids(tok, E.mbpp_prompt(p)) for p in probs]
    t = time.time()
    outs = E.sample(model, tok, prompts, a.k, 160, 1.0, seed=123)
    texts = [tok.decode(x, skip_special_tokens=True) for o in outs for x in o]
    g = E.grade_many(texts, [p for p in probs for _ in range(a.k)])
    res = {str(p["task_id"]): sum(g[i * a.k:(i + 1) * a.k]) for i, p in enumerate(probs)}
    json.dump(dict(k=a.k, passes=res), open(os.path.join(a.out, "screen.json"), "w"))
    hit = sum(v > 0 for v in res.values())
    E.log(f"screen: {len(probs)} problems x {a.k} samples in {time.time() - t:.0f}s; passes {sum(g)}/{len(g)}; "
          f"problems with >=1 pass {hit}")


def teacher(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ttok = AutoTokenizer.from_pretrained(TEACHER)
    tm = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).eval()
    stok, _ = E.load()
    probs = E.problems("train")
    prompts = []
    for p in probs:
        txt = ttok.apply_chat_template([{"role": "user", "content": E.mbpp_prompt(p)}], tokenize=False, add_generation_prompt=True)
        prompts.append(ttok(txt, add_special_tokens=False)["input_ids"])
    t = time.time()
    outs = E.sample(tm, ttok, prompts, 4, 256, 0.7, seed=7, rows_per_chunk=48)
    bank, n_ok = {}, 0
    for p, o in zip(probs, outs):
        texts = [ttok.decode(x, skip_special_tokens=True) for x in o]
        g = E.grade_many(texts, [p] * len(texts))
        sols = []
        for txt, ok in zip(texts, g):
            code = E.extract_code(txt).strip()
            if ok and code not in sols:
                sols.append(code)
        bank[str(p["task_id"])] = sols[:2]
        n_ok += bool(sols)
    json.dump(bank, open(os.path.join(a.out, "teacher.json"), "w"), indent=0)
    E.log(f"teacher: {n_ok}/{len(probs)} problems have a test-passing teacher solution ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["screen", "teacher"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    dict(screen=screen, teacher=teacher)[a.what](a)
