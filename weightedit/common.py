"""Shared helpers for the Qwen3.5-0.8B weight-editing experiment: problems, prompts, answer checking, loop detection."""
import json, os, random, re

PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_problems(split, n, seed=0, cache=None):
    """split: 'practice' (GSM8K train + MATH train, never the test sets) or 'math500' (HuggingFaceH4/MATH-500 test)."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    get = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset", local_dir=cache)
    rng = random.Random(seed)
    if split == "math500":
        rows = [json.loads(l) for l in open(get("HuggingFaceH4/MATH-500", "test.jsonl")) if l.strip()]
        rng.shuffle(rows)
        return [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"], "source": "math500"} for r in rows[:n]]
    gsm = pq.read_table(get("openai/gsm8k", "main/train-00000-of-00001.parquet")).to_pylist()
    math = []
    for sub in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"):
        math += pq.read_table(get("EleutherAI/hendrycks_math", f"{sub}/train-00000-of-00001.parquet")).to_pylist()
    rng.shuffle(gsm)
    rng.shuffle(math)
    out = [{"id": f"gsm-{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip(), "source": "gsm8k"}
           for i, r in enumerate(gsm[:n // 2])]
    out += [{"id": f"math-{i}", "problem": r["problem"], "answer": last_boxed(r["solution"]) or "", "source": "math_train"}
            for i, r in enumerate(math[:n - n // 2])]
    rng.shuffle(out)
    return out


def chat_prompt(tok, problem):
    return tok.apply_chat_template([{"role": "user", "content": problem + PROMPT_SUFFIX}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def last_boxed(text):
    """Content of the last \\boxed{...} (balanced braces), or None."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k].strip()
    return None


def _norm(s):
    s = s.replace("\\!", "").replace("\\left", "").replace("\\right", "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = s.replace("$", "").replace(" ", "").replace(",", "").rstrip(".")
    return s


def is_correct(pred, gold):
    """math_verify when available (robust symbolic comparison), else normalised string match."""
    if pred is None:
        return False
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        return _norm(pred) == _norm(gold)


def loop_score(token_ids, n=32, tail=2048):
    """Share of the last `tail` tokens' n-grams that already occurred earlier in that tail (0 = no repetition)."""
    t = token_ids[-tail:]
    seen, rep, total = set(), 0, 0
    for i in range(len(t) - n + 1):
        g = tuple(t[i:i + n])
        total += 1
        rep += g in seen
        seen.add(g)
    return rep / max(total, 1)
