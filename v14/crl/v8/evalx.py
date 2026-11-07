#!/usr/bin/env python3
"""Score one model folder on all 500 MBPP test problems and all 164 HumanEval problems:
  sampled pass@1 = share of --n answers (T=0.8, the same seeds for every model) that pass, averaged over problems,
  greedy pass@1  = one greedy answer per problem.
Per-problem pass counts are saved so two models can be compared problem by problem (compare.py).

  python -I evalx.py --model DIR --out FILE.json
"""
import argparse, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402


def he_grade(text, p):
    ok, _ = E.run_program(E.build_program(p["prompt"], p["entry_point"], E.extract_code(text), p["test"]), timeout=5)
    return int(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N problems of each set (smoke tests)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    tok, model = E.load(a.model, dtype=torch.bfloat16)
    he = pq.read_table(hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet",
                                       repo_type="dataset")).to_pylist()
    mt = E.problems("test")
    if a.limit:
        he, mt = he[:a.limit], mt[:a.limit]
    res = {}
    t0 = time.time()
    for name, probs, prompt_of, grader in (
            ("mbpp_test", mt, lambda p: E.mbpp_prompt(p), lambda t, p: E.grade(t, p)),
            ("humaneval", he, lambda p: E.HE_INSTR.format(prompt=p["prompt"]), he_grade)):
        prompts = [E.chat_ids(tok, prompt_of(p)) for p in probs]
        sampled = E.sample(model, tok, prompts, a.n, 256, 0.8, seed=2024)
        greedy = E.sample(model, tok, prompts, 1, 256, 0, seed=0)
        jobs = [(tok.decode(x, skip_special_tokens=True), p) for p, o in zip(probs, sampled) for x in o]
        jobs += [(tok.decode(o[0], skip_special_tokens=True), p) for p, o in zip(probs, greedy)]
        with ThreadPoolExecutor(4) as ex:
            g = list(ex.map(lambda j: grader(*j), jobs))
        ns = len(probs) * a.n
        per_s = [sum(g[i * a.n:(i + 1) * a.n]) for i in range(len(probs))]
        per_g = g[ns:]
        res[name] = dict(sampled_pass1=sum(per_s) / ns, greedy_pass1=sum(per_g) / len(probs), n=a.n, problems=len(probs),
                         per_sampled=per_s, per_greedy=per_g)
        E.log(f"{name}: sampled pass@1 {res[name]['sampled_pass1']:.4f}  greedy {sum(per_g)}/{len(probs)}  ({time.time() - t0:.0f}s)")
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
