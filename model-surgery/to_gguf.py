#!/usr/bin/env python3
"""Step 6: convert an HF model dir to GGUF and quantize it (Q6_K, Q8_0, Q4_K_M, ...).

Needs a llama.cpp source tree (for convert_hf_to_gguf.py) and llama-cpp-python (for quantizing and the
load test). Both come from one place: `pip download llama-cpp-python --no-binary :all:`, extract it, and
point --llama-cpp at <extracted>/vendor/llama.cpp; then `pip install` that extracted dir.

Handles two problems hit in practice:
  * vocab-pruned tokenizers have a new hash that the converter doesn't recognise. Pass
    --pre-tokenizer-from ORIGINAL_DIR (the unpruned model) and the original's pre-tokenizer name is reused;
    pruning never changes the pre-tokenizer rules, only the vocab.
  * text-only copies of models with an MTP/NextN head: --no-mtp is added automatically when the
    config says there are 0 MTP layers and the converter supports the flag.

  python -I to_gguf.py --model DIR --llama-cpp LLAMA_CPP_DIR --out-dir OUT --quants Q6_K Q8_0 \
      [--pre-tokenizer-from ORIGINAL_DIR] [--keep-bf16]
"""
import argparse, json, os, runpy, subprocess, sys, tempfile, time

HERE = os.path.abspath(__file__)


def run_converter(llama_cpp, args, forced_pre=None, record_to=None):
    """Run convert_hf_to_gguf.py in-process so its pre-tokenizer lookup can be wrapped."""
    sys.path.insert(0, llama_cpp)
    sys.path.insert(0, os.path.join(llama_cpp, "gguf-py"))
    import conversion.base as base
    orig = base.TextModel.get_vocab_base_pre

    def wrapped(self, tokenizer):
        try:
            res = orig(self, tokenizer)
        except NotImplementedError:
            if not forced_pre:
                raise
            print(f"[to_gguf] unrecognised pre-tokenizer hash; using {forced_pre!r} from the original model", file=sys.stderr)
            res = forced_pre
        if record_to:
            open(record_to, "w").write(res)
        return res

    base.TextModel.get_vocab_base_pre = wrapped
    sys.argv = [os.path.join(llama_cpp, "convert_hf_to_gguf.py")] + args
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise


def subrun(spec):
    """Each conversion gets a fresh interpreter (the converter keeps class-level state)."""
    r = subprocess.run([sys.executable, "-I", HERE, "_convert", json.dumps(spec)])
    if r.returncode != 0:
        raise SystemExit(f"conversion failed: {spec}")


def needs_no_mtp(model_dir, llama_cpp):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    tc = cfg.get("text_config") or {}
    keys = ("mtp_num_hidden_layers", "num_nextn_predict_layers")
    zero = any(k in d and not d[k] for d in (cfg, tc) for k in keys)
    supported = "--no-mtp" in open(os.path.join(llama_cpp, "convert_hf_to_gguf.py")).read()
    return zero and supported


def quantize(src, dst, qtype, threads):
    import ctypes
    import llama_cpp
    ftype = getattr(llama_cpp, f"LLAMA_FTYPE_MOSTLY_{qtype}", None)
    if ftype is None:
        raise SystemExit(f"unknown quant type {qtype}")
    p = llama_cpp.llama_model_quantize_default_params()
    p.ftype = ftype
    p.nthread = threads
    rc = llama_cpp.llama_model_quantize(src.encode(), dst.encode(), ctypes.byref(p))
    if rc != 0:
        raise SystemExit(f"quantize {qtype} failed (rc={rc})")


def smoke(gguf, threads):
    from llama_cpp import Llama
    llm = Llama(model_path=gguf, n_ctx=1024, n_threads=threads, verbose=False)
    msgs = [{"role": "user", "content": "Write a Python function that returns the sum of a list."}]
    t = time.time()
    r = llm.create_chat_completion(msgs, max_tokens=96, temperature=0)
    s = time.time() - t
    n = r["usage"]["completion_tokens"]
    return {"tok_per_s": round(n / s, 1), "reply": r["choices"][0]["message"]["content"][:300]}


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "_convert":
        spec = json.loads(sys.argv[2])
        run_converter(spec["llama_cpp"], spec["args"], spec.get("forced_pre"), spec.get("record_to"))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp source dir containing convert_hf_to_gguf.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", help="file name stem (default: model dir name)")
    ap.add_argument("--quants", nargs="*", default=["Q6_K"])
    ap.add_argument("--pre-tokenizer-from", help="unpruned original model dir, to reuse its pre-tokenizer name")
    ap.add_argument("--keep-bf16", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    stem = a.name or os.path.basename(os.path.normpath(a.model))
    extra = ["--no-mtp"] if needs_no_mtp(a.model, a.llama_cpp) else []
    forced = None
    if a.pre_tokenizer_from:
        with tempfile.TemporaryDirectory() as d:
            rec = os.path.join(d, "pre.txt")
            subrun({"llama_cpp": a.llama_cpp, "record_to": rec,
                    "args": [a.pre_tokenizer_from, "--vocab-only", "--outfile", os.path.join(d, "v.gguf")]
                    + (["--no-mtp"] if needs_no_mtp(a.pre_tokenizer_from, a.llama_cpp) else [])})
            forced = open(rec).read().strip()
        print(f"original pre-tokenizer: {forced}")
    bf16 = os.path.join(a.out_dir, f"{stem}-bf16.gguf")
    subrun({"llama_cpp": a.llama_cpp, "forced_pre": forced,
            "args": [a.model, "--outtype", "bf16", "--outfile", bf16] + extra})
    report = {"bf16": bf16, "no_mtp": bool(extra), "pre_tokenizer_forced": forced, "quants": {}}
    for q in a.quants:
        out = os.path.join(a.out_dir, f"{stem}-{q}.gguf")
        quantize(bf16, out, q, a.threads)
        report["quants"][q] = {"path": out, "size_GB": round(os.path.getsize(out) / 1e9, 3), **smoke(out, a.threads)}
        print(q, report["quants"][q])
    if not a.keep_bf16 and a.quants:
        os.remove(bf16)
        report["bf16"] = None
    json.dump(report, open(os.path.join(a.out_dir, f"{stem}-gguf.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
