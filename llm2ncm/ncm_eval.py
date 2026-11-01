#!/usr/bin/env python3
"""Stage D: how much of the original LLM survived the conversion? Always side by side with the original.

On held-out code and coding-chat prompts, for the original LLM (normal token-by-token generation with
KV cache) and the NCM at each --refine setting:
  * judge bits/token  - the ORIGINAL model's surprise at the output (lower = more like what it would write);
                        the real held-out continuation is scored too, as a reference point
  * agreement         - share of positions matching the original model's greedy output
  * distinct-2, repeated lines - degeneration checks
  * speed             - tokens/s and forward passes per token, same PyTorch runtime for both
  * HumanEval (opt.)  - executed tests, grader reused from model-surgery/evaluate.py

  python -I ncm_eval.py --model DIR --generator work/generator --data msdata/ --out work/eval.json
"""
import argparse, contextlib, importlib.util, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
from ncm_io import end_ids, load_generator


def surgery_eval():
    path = os.path.join(HERE, "..", "model-surgery", "evaluate.py")
    spec = importlib.util.spec_from_file_location("ms_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path, n):
    out = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            out.append(json.loads(line))
        if len(out) >= n:
            break
    return out


def distinct2(outs):
    grams = [tuple(o[i:i + 2]) for o in outs for i in range(len(o) - 1)]
    return len(set(grams)) / max(len(grams), 1)


def repeated_lines(texts):
    rep = tot = 0
    for t in texts:
        seen = set()
        for line in (l.strip() for l in t.splitlines()):
            if len(line) < 4:
                continue
            tot += 1
            rep += line in seen
            seen.add(line)
    return rep / max(tot, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--data", required=True, help="dir with heldout_code.jsonl, heldout_chat.jsonl, humaneval.parquet")
    ap.add_argument("--out")
    ap.add_argument("--n-prompts", type=int, default=16, help="half code, half chat")
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--refine", default="1,2,4")
    ap.add_argument("--humaneval", type=int, default=0, help="number of HumanEval problems (0 = skip)")
    ap.add_argument("--he-max-new", type=int, default=256)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    tok, gen, comp, cfg = load_generator(a.model, a.generator, a.threads)
    lm = gen.lm
    teacher = lm.disable_adapter if hasattr(lm, "disable_adapter") else contextlib.nullcontext
    stops = end_ids(tok)
    P = cfg["compressor_cfg"]["P"]
    refines = [int(r) for r in a.refine.split(",")]

    prompts = []  # (kind, prefix ids, real continuation ids)
    for r in read_jsonl(os.path.join(a.data, "heldout_code.jsonl"), 400):
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= P + a.new and len(prompts) < a.n_prompts // 2:
            prompts.append(("code", ids[:P], ids[P:P + a.new]))
    for r in read_jsonl(os.path.join(a.data, "heldout_chat.jsonl"), 200):
        if len(prompts) >= a.n_prompts:
            break
        head = tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False, add_generation_prompt=True)
        prompts.append(("chat", tok(head, add_special_tokens=False)["input_ids"],
                        tok(r["response"], add_special_tokens=False)["input_ids"][:a.new]))

    def judge_bits(prefix, cont):
        if not cont:
            return None
        with teacher(), torch.inference_mode():
            ids = torch.tensor([prefix + cont])
            logits = lm(input_ids=ids).logits[0, len(prefix) - 1:-1].float()
            nll = torch.nn.functional.cross_entropy(logits, torch.tensor(cont), reduction="sum")
        return float(nll) / 0.6931471805599453, len(cont)

    def ar_generate(prefix, n):
        with teacher(), torch.inference_mode():
            out = lm.generate(torch.tensor([prefix]), attention_mask=torch.ones(1, len(prefix), dtype=torch.long),
                              max_new_tokens=n, do_sample=False, eos_token_id=sorted(stops),
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = out[0, len(prefix):].tolist()
        return new[:next((i for i, t in enumerate(new) if t in stops), len(new))]

    systems = {"original (token-by-token)": None}
    for R in refines:
        systems[f"NCM refine={R}"] = R
    results, outputs = {}, {}
    ar_out = {}
    for name, R in systems.items():
        t0 = time.time()
        outs, passes = [], 0
        for i, (_, prefix, _) in enumerate(prompts):
            if R is None:
                o = ar_generate(prefix, a.new)
                passes += len(o) + 1
                ar_out[i] = o
            else:
                o, p = gen.generate(comp.rvq, prefix, a.new, refine=R, end_ids=stops)
                passes += p
            outs.append(o)
        secs = time.time() - t0
        ntok = sum(len(o) for o in outs)
        bits = [judge_bits(p[1], o) for p, o in zip(prompts, outs)]
        bits = [b for b in bits if b]
        agree = [sum(x == y for x, y in zip(o, ar_out[i])) / max(len(ar_out[i]), 1) for i, o in enumerate(outs)]
        texts = [tok.decode(o) for o in outs]
        results[name] = {"judge_bits_per_token": round(sum(b for b, _ in bits) / max(sum(n for _, n in bits), 1), 3),
                         "agreement_with_original": round(sum(agree) / len(agree), 3),
                         "distinct2": round(distinct2(outs), 3), "repeated_lines": round(repeated_lines(texts), 3),
                         "tokens_per_s": round(ntok / secs, 1), "passes_per_token": round(passes / max(ntok, 1), 3),
                         "tokens": ntok}
        outputs[name] = texts[:4]
        print(name, results[name], flush=True)
    real = [judge_bits(p[1], p[2]) for p in prompts]
    real = [b for b in real if b]
    results["real held-out text (reference)"] = {"judge_bits_per_token": round(sum(b for b, _ in real) / sum(n for _, n in real), 3)}

    if a.humaneval:
        ms = surgery_eval()
        probs = ms.load_humaneval(os.path.join(a.data, "humaneval.parquet"), a.humaneval)
        best = max(refines)
        for name, R in (("original (token-by-token)", None), (f"NCM refine={best}", best)):
            passed = 0
            for p in probs:
                ids = tok(ms.chat_prompt(tok, ms.HE_INSTR.format(prompt=p["prompt"])), add_special_tokens=False)["input_ids"]
                o = ar_generate(ids, a.he_max_new) if R is None else gen.generate(comp.rvq, ids, a.he_max_new, refine=R, end_ids=stops)[0]
                ok, _ = ms.run_program(ms.build_program(p["prompt"], p["entry_point"], ms.extract_code(tok.decode(o, skip_special_tokens=True)), p["test"]))
                passed += ok
            results[name]["humaneval"] = f"{passed}/{len(probs)}"
            print(name, "HumanEval", results[name]["humaneval"], flush=True)

    rows = ["| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |",
            "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        rows.append(f"| {name} | {r['judge_bits_per_token']} | {r.get('agreement_with_original', '-')} | {r.get('distinct2', '-')} | "
                    f"{r.get('repeated_lines', '-')} | {r.get('tokens_per_s', '-')} | {r.get('passes_per_token', '-')} | {r.get('humaneval', '-')} |")
    table = "\n".join(rows)
    print(table)
    if a.out:
        json.dump({"results": results, "table": table, "samples": outputs, "prompts": len(prompts), "new_tokens": a.new,
                   "generator": cfg}, open(a.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
