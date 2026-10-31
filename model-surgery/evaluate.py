#!/usr/bin/env python3
"""Step 5: measure a model. Use this as a gate after every change - lesson learned: bits/byte can look
fine while the model can no longer write working code, so always run some HumanEval.

Metrics
  --text NAME=PATH.jsonl[:FIELD]   bits per UTF-8 byte of plain text (lower is better; tokenizer-independent)
  --chat NAME=PATH.jsonl[:USER:ASSISTANT]  bits/byte of assistant replies given the chat-rendered prompt
  --humaneval PATH.parquet         pass@1, greedy, code executed against the tests (subprocess, timeout)

Backends: PyTorch (MODEL is an HF dir) or llama.cpp (MODEL is a .gguf; pass --hf-dir for the tokenizer).

  python -I evaluate.py MODEL --out result.json --text code=heldout.jsonl --humaneval he.parquet --limit-humaneval 20
  python -I evaluate.py model.Q6_K.gguf --hf-dir HF_DIR --out q6.json --humaneval he.parquet
  python -I evaluate.py --check-canonical --humaneval he.parquet     # self-test the grader, no model
"""
import argparse, json, math, os, re, resource, subprocess, sys, tempfile, time

HE_INSTR = ("Complete the following Python function. Respond with the full function in a single "
            "```python code block.\n\n```python\n{prompt}```")
END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>", "<|end|>", "</s>")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ HumanEval grading

def extract_code(text):
    m = re.search(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    i = text.find("```")
    if i >= 0:
        return re.sub(r"^(?:python|py|Python)?[ \t]*\n", "", text[i + 3:], count=1)
    return text


def build_program(prompt, entry_point, code, test):
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.M):
        # full function returned: keep the prompt's imports and any helper defs it relies on
        helpers = prompt.split(f"def {entry_point}")[0]
        program = helpers + "\n" + code
    else:
        program = prompt + code
    return program + "\n\n" + test + f"\n\ncheck({entry_point})\n"


def run_program(program, timeout=10):
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    with tempfile.TemporaryDirectory(prefix="he_") as d:
        path = os.path.join(d, "prog.py")
        open(path, "w", encoding="utf-8").write(program)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, errors="replace",
                               timeout=timeout, preexec_fn=limits, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if p.returncode == 0:
            return True, "pass"
        lines = [l for l in p.stderr.strip().splitlines() if l.strip()]
        return False, (lines[-1][:300] if lines else f"exit {p.returncode}")


def load_humaneval(path, limit=None):
    import pyarrow.parquet as pq
    probs = pq.read_table(path).to_pylist()
    probs.sort(key=lambda p: int(p["task_id"].split("/")[1]))
    return probs[:limit] if limit else probs


# ------------------------------------------------------------------ backends

class TorchLM:
    def __init__(self, model_dir, dtype, threads):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        gc = getattr(self.model, "generation_config", None)
        self.extra_eos = gc.eos_token_id if gc and gc.eos_token_id is not None else []

    def nll_bits(self, ctx, tgt):
        torch = self.torch
        ids = torch.tensor([ctx + tgt])
        with torch.inference_mode():
            lg = self.model(input_ids=ids, use_cache=False).logits[0].float()
        pred = lg[len(ctx) - 1: len(ctx) - 1 + len(tgt)]
        return torch.nn.functional.cross_entropy(pred, torch.tensor(tgt), reduction="sum").item() / math.log(2)

    def generate(self, prompts, max_new, eos, batch):
        torch = self.torch
        tok = self.tok
        tok.padding_side = "left"
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
        outs = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            with torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=eos, pad_token_id=pad)
            for j, i in enumerate(idx):
                outs[i] = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            log(f"  generated {min(s + batch, len(order))}/{len(order)}")
        return outs


class LlamaCppLM:
    def __init__(self, gguf, hf_dir, threads, n_ctx=4096):
        import numpy as np
        from llama_cpp import Llama
        from transformers import AutoTokenizer
        self.np = np
        self.tok = AutoTokenizer.from_pretrained(hf_dir)
        self.llm = Llama(model_path=gguf, n_ctx=n_ctx, n_threads=threads, n_threads_batch=threads,
                         logits_all=True, n_batch=512, verbose=False)
        self.n_params = self.llm.n_params() if hasattr(self.llm, "n_params") else None
        self.extra_eos = [self.llm.token_eos()]

    def nll_bits(self, ctx, tgt):
        np = self.np
        self.llm.reset()
        self.llm.eval(list(ctx) + list(tgt))
        lg = np.asarray(self.llm.scores[len(ctx) - 1: len(ctx) - 1 + len(tgt)], dtype=np.float64)
        m = lg.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(lg - m).sum(1))
        return float((lse - lg[np.arange(len(tgt)), tgt]).sum()) / math.log(2)

    def generate(self, prompts, max_new, eos, batch):
        outs = []
        for k, p in enumerate(prompts):
            ids = self.tok(p, add_special_tokens=False)["input_ids"]
            self.llm.reset()
            gen = []
            for t in self.llm.generate(ids, top_k=1, top_p=1.0, temp=0.0, repeat_penalty=1.0, reset=True):
                if t in eos or len(gen) >= max_new:
                    break
                gen.append(t)
            outs.append(self.tok.decode(gen, skip_special_tokens=True))
            log(f"  generated {k + 1}/{len(prompts)}")
        return outs


def end_ids(lm):
    tok = lm.tok
    ids = set()
    for t in END_TOKENS:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id and i is not None:
            ids.add(i)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    extra = lm.extra_eos
    ids.update(extra if isinstance(extra, list) else [extra])
    return sorted(ids)


def chat_prompt(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


# ------------------------------------------------------------------ metrics

def parse_spec(spec, default_fields):
    name, rest = spec.split("=", 1)
    path, _, fields = rest.partition(":") if not os.path.exists(rest) else (rest, "", "")
    return name, path, (fields.split(":") if fields else default_fields)


def read_jsonl(path, limit):
    out = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return out[:limit] if limit else out


def score_text(lm, recs, field):
    tok = lm.tok
    ctx = [tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id]
    bits = nbytes = ntok = 0
    t0 = time.time()
    for r in recs:
        tgt = tok(r[field], add_special_tokens=False)["input_ids"]
        bits += lm.nll_bits(ctx, tgt); nbytes += len(r[field].encode()); ntok += len(tgt)
    return {"bits_per_byte": bits / nbytes, "texts": len(recs), "tokens": ntok, "tokens_per_s": round(ntok / (time.time() - t0), 1)}


def score_chat(lm, recs, uf, af):
    tok = lm.tok
    bits = nbytes = ntok = 0
    t0 = time.time()
    for r in recs:
        ctx = tok(chat_prompt(tok, r[uf]), add_special_tokens=False)["input_ids"]
        tgt = tok(r[af], add_special_tokens=False)["input_ids"]
        bits += lm.nll_bits(ctx, tgt); nbytes += len(r[af].encode()); ntok += len(tgt)
    return {"bits_per_byte": bits / nbytes, "texts": len(recs), "tokens": ntok, "tokens_per_s": round(ntok / (time.time() - t0), 1)}


def humaneval(lm, probs, max_new, batch):
    eos = end_ids(lm)
    prompts = [chat_prompt(lm.tok, HE_INSTR.format(prompt=p["prompt"])) for p in probs]
    t0 = time.time()
    comps = lm.generate(prompts, max_new, eos, batch)
    gen_s = time.time() - t0
    per, passed = [], 0
    for p, c in zip(probs, comps):
        ok, why = run_program(build_program(p["prompt"], p["entry_point"], extract_code(c), p["test"]))
        passed += ok
        per.append({"task_id": p["task_id"], "passed": ok, "reason": why, "completion": c})
    return {"pass_at_1": passed / len(probs), "passed": passed, "total": len(probs),
            "generation_seconds": round(gen_s, 1), "per_problem": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?")
    ap.add_argument("--hf-dir", help="tokenizer/chat-template dir (required for .gguf)")
    ap.add_argument("--out")
    ap.add_argument("--text", action="append", default=[], help="NAME=PATH.jsonl[:FIELD] (default field text)")
    ap.add_argument("--chat", action="append", default=[], help="NAME=PATH.jsonl[:USER:ASSISTANT] (default instruction:response)")
    ap.add_argument("--humaneval", help="HumanEval parquet (openai/openai_humaneval)")
    ap.add_argument("--limit-texts", type=int)
    ap.add_argument("--limit-humaneval", type=int)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="fp32 is usually faster on CPUs without bf16 units")
    ap.add_argument("--check-canonical", action="store_true")
    a = ap.parse_args()

    if a.check_canonical:
        probs = load_humaneval(a.humaneval)
        bad = [p["task_id"] for p in probs if not run_program(build_program(
            p["prompt"], p["entry_point"], p["prompt"] + p["canonical_solution"], p["test"]))[0]]
        print(f"canonical solutions: {len(probs) - len(bad)}/{len(probs)} pass", bad)
        sys.exit(1 if bad else 0)
    if not a.model:
        ap.error("model required")

    t = time.time()
    if a.model.endswith(".gguf"):
        if not a.hf_dir:
            ap.error("--hf-dir is required for .gguf models")
        lm = LlamaCppLM(a.model, a.hf_dir, a.threads)
    else:
        lm = TorchLM(a.model, a.dtype, a.threads)
    res = {"model": a.model, "load_seconds": round(time.time() - t, 1),
           "params_M": round(lm.n_params / 1e6, 1) if lm.n_params else None, "metrics": {}}
    log("loaded", a.model)
    for spec in a.text:
        name, path, fields = parse_spec(spec, ["text"])
        log("text", name); res["metrics"][name] = score_text(lm, read_jsonl(path, a.limit_texts), fields[0])
    for spec in a.chat:
        name, path, fields = parse_spec(spec, ["instruction", "response"])
        log("chat", name); res["metrics"][name] = score_chat(lm, read_jsonl(path, a.limit_texts), fields[0], fields[1])
    if a.humaneval:
        log("humaneval"); res["metrics"]["humaneval"] = humaneval(lm, load_humaneval(a.humaneval, a.limit_humaneval), a.max_new, a.batch)
    summary = {k: {m: v for m, v in d.items() if m != "per_problem"} for k, d in res["metrics"].items()}
    print(json.dumps(summary, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
