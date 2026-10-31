#!/usr/bin/env python3
"""One command: specialize + shrink + quantize a model, with HumanEval gates so a broken model never ships.

  1. textonly      drop vision tower / MTP head                         (lossless)
  2. counts        token frequencies on the domain counting data
  3. prune_vocab   keep domain tokens within an inflation budget        (lossless on kept-token text)
  4. verify        tokenization + logits + generation check             (must pass)
  5. gate          HumanEval subset + bits/byte: pruned vs text-only    (must not drop more than allowed)
  6. to_gguf       bf16 GGUF -> quants (default Q6_K)
  7. gate          same HumanEval subset on each quant via llama.cpp
Every step's output lives under --work and is reused if present, so a crashed run resumes.
Writes --work/REPORT.md and REPORT.json.

  python -I fetch_data.py --out data/                      # once, for the "code" domain
  python -I pipeline.py --model Qwen/Qwen3.5-2B --data data/ --work runs/qwen35-2b \
      --llama-cpp /path/to/llama.cpp [--quants Q6_K Q8_0] [--max-inflation 0.015]
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(step, *args):
    cmd = [sys.executable, "-I", os.path.join(HERE, step)] + [str(x) for x in args]
    print("\n$", " ".join(cmd), flush=True)
    t = time.time()
    r = subprocess.run(cmd)
    print(f"[{step} exit {r.returncode}, {time.time() - t:.0f}s]", flush=True)
    return r.returncode


def must(step, *args):
    if run(step, *args) != 0:
        raise SystemExit(f"step {step} failed; see output above")


def evaluate(model, out, data, a, hf_dir=None):
    if os.path.exists(out):
        return json.load(open(out))
    args = [model, "--out", out, "--threads", a.threads, "--limit-texts", a.gate_texts,
            "--text", f"code={data}/heldout_code.jsonl", "--chat", f"chat={data}/heldout_chat.jsonl",
            "--text", f"general={data}/heldout_general.jsonl",
            "--humaneval", f"{data}/humaneval.parquet", "--limit-humaneval", a.gate_problems]
    if hf_dir:
        args += ["--hf-dir", hf_dir]
    must("evaluate.py", *args)
    return json.load(open(out))


def gate(name, base, cand, a, notes):
    b = base["metrics"]["humaneval"]["passed"]
    c = cand["metrics"]["humaneval"]["passed"]
    ok = c >= b - a.max_pass_drop
    msg = f"{name}: HumanEval {c}/{a.gate_problems} vs baseline {b}/{a.gate_problems} -> {'PASS' if ok else 'FAIL'}"
    print("GATE", msg, flush=True)
    notes.append(msg)
    return ok


def row(name, res, size=None, speed=None):
    m = res["metrics"]
    he = m["humaneval"]
    f = lambda k: f"{m[k]['bits_per_byte']:.4f}" if k in m else "-"
    return f"| {name} | {res.get('params_M') or '-'} | {size or '-'} | {he['passed']}/{he['total']} | {f('code')} | {f('chat')} | {f('general')} | {speed or '-'} |"


def dir_size(p):
    return f"{sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / 1e9:.2f} GB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local dir")
    ap.add_argument("--data", required=True, help="dir from fetch_data.py (or same file layout)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--llama-cpp", help="llama.cpp source dir; omit to skip GGUF/quantization")
    ap.add_argument("--quants", nargs="*", default=["Q6_K"])
    ap.add_argument("--max-inflation", type=float, default=0.015)
    ap.add_argument("--skip-vocab", action="store_true")
    ap.add_argument("--gate-problems", type=int, default=20, help="HumanEval problems per gate (more = slower, less noisy)")
    ap.add_argument("--gate-texts", type=int, default=40)
    ap.add_argument("--max-pass-drop", type=int, default=1, help="allowed HumanEval regressions vs the text-only baseline")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    W, D = a.work, a.data
    os.makedirs(W, exist_ok=True)
    notes, rows = [], []

    t1 = f"{W}/1-textonly"
    if not os.path.exists(f"{t1}/config.json"):
        must("textonly.py", a.model, t1)
    base = evaluate(t1, f"{W}/eval-textonly.json", D, a)
    rows.append(row("text-only", base, dir_size(t1)))
    final = t1

    t2 = f"{W}/2-vocab"
    if not a.skip_vocab:
        counts = f"{W}/counts.npy"
        if not os.path.exists(counts):
            must("count_tokens.py", "--model", t1, "--out", counts,
                 "--jsonl", f"{D}/count_code.jsonl:text", "--chat-jsonl", f"{D}/count_chat.jsonl:instruction:response")
        if not os.path.exists(f"{t2}/vocab_prune.json"):
            rc = run("prune_vocab.py", "--src", t1, "--dst", t2, "--counts", counts, "--max-inflation", a.max_inflation,
                     "--heldout", f"{D}/heldout_code.jsonl:text", "--heldout", f"{D}/heldout_chat.jsonl:instruction,response")
            if rc != 0:
                notes.append("vocab pruning skipped: tokenizer not supported or pruning failed (see log)")
        if os.path.exists(f"{t2}/vocab_prune.json"):
            if not os.path.exists(f"{W}/verify.json"):
                must("verify.py", "--orig", t1, "--pruned", t2, "--heldout", f"{D}/heldout_code.jsonl:text",
                     "--out", f"{W}/verify.json", "--threads", a.threads)
            notes.append(f"verify: {json.dumps({k: v for k, v in json.load(open(f'{W}/verify.json')).items() if k in ('tokenization', 'logits', 'ok')})}")
            pr = evaluate(t2, f"{W}/eval-vocab.json", D, a)
            rows.append(row("vocab-pruned", pr, dir_size(t2)))
            if gate("vocab-pruned", base, pr, a, notes):
                final = t2
            else:
                notes.append("vocab-pruned model rejected; continuing with the text-only model")

    if a.llama_cpp:
        G = f"{W}/gguf"
        stem = "model"
        rep = f"{G}/{stem}-gguf.json"
        if not os.path.exists(rep):
            args = ["--model", final, "--llama-cpp", a.llama_cpp, "--out-dir", G, "--name", stem, "--threads", a.threads, "--quants", *a.quants]
            if final == t2:
                args += ["--pre-tokenizer-from", t1]
            must("to_gguf.py", *args)
        g = json.load(open(rep))
        for q, info in g["quants"].items():
            r = evaluate(info["path"], f"{W}/eval-{q}.json", D, a, hf_dir=final)
            rows.append(row(f"{os.path.basename(final)} {q}", r, f"{info['size_GB']:.2f} GB", f"{info['tok_per_s']} tok/s"))
            gate(q, base, r, a, notes)

    hdr = ["| model | params (M) | size | HumanEval | code bits/byte | chat bits/byte | general bits/byte | gen speed |",
           "|---|---|---|---|---|---|---|---|"]
    report = "\n".join([f"# Surgery report: {a.model}", "", *hdr, *rows, "", "Lower bits/byte is better. "
                        f"HumanEval = first {a.gate_problems} problems, greedy, code executed against tests.",
                        "", "## Notes", *[f"- {n}" for n in notes], "", f"Final HF model: `{final}`"])
    open(f"{W}/REPORT.md", "w").write(report + "\n")
    json.dump({"final_hf": final, "notes": notes, "rows": rows}, open(f"{W}/REPORT.json", "w"), indent=2)
    print("\n" + report)


if __name__ == "__main__":
    main()
