#!/usr/bin/env python3
"""Generate with a converted NCM (chat template applied unless --raw).

  python -I ncm_generate.py --model DIR --generator work/generator --prompt "Write a Python function that ..." --refine 4
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncm_io import end_ids, load_generator, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--refine", type=int, default=4, help="token-level passes per block (more = better, slower)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--raw", action="store_true", help="treat --prompt as plain text, no chat template")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    a = ap.parse_args()
    tok, gen, comp, _ = load_generator(a.model, a.generator, a.threads, pick_device(a.device))
    text = a.prompt if a.raw else tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    t = time.time()
    out, passes = gen.generate(comp.rvq, ids, a.new, refine=a.refine, temperature=a.temperature, end_ids=end_ids(tok))
    s = time.time() - t
    print(tok.decode(out, skip_special_tokens=True))
    print(f"\n[{len(out)} tokens, {passes} forward passes, {len(out) / s:.1f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
