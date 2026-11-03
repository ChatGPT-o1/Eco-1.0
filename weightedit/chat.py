#!/usr/bin/env python3
"""Ask a model a few questions (greedy, so two models can be compared line by line).

  python -I chat.py --model DIR [--think] [--max-new 200] [--q "your question" ...]
"""
import argparse, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = ["What is 2 + 2?", "What is the capital of France?", "Write a Python function that adds two numbers."]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", action="store_true", help="thinking mode (long answers); default is the short answer mode")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--q", action="append")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32 if a.device == "cpu" else torch.float16).to(a.device).eval()
    for q in a.q or QUESTIONS:
        p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=a.think)
        enc = tok(p, return_tensors="pt").to(a.device)
        t = time.time()
        with torch.inference_mode():
            out = lm.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n### {q}   ({out.shape[1] - enc['input_ids'].shape[1]} tokens, {time.time() - t:.0f}s)\n{text.strip()}", flush=True)


if __name__ == "__main__":
    main()
