#!/usr/bin/env python3
"""Compare evalx.py results problem by problem: differences in sampled pass@1 with 95% bootstrap intervals over problems.

  python -I compare.py base.json grpo.json v8.json
"""
import json, random, sys


def boot(d, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    ms = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def main():
    files = sys.argv[1:]
    R = {f: json.load(open(f)) for f in files}
    names = [f.rsplit("/", 1)[-1].replace(".json", "") for f in files]
    for task in ("mbpp_test", "humaneval"):
        if not all(task in R[f] for f in files):
            continue
        print(f"\n{task}")
        for f, nm in zip(files, names):
            r = R[f][task]
            print(f"  {nm:10s} sampled pass@1 {r['sampled_pass1']:.2%}   greedy {sum(r['per_greedy'])}/{r['problems']}")
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                a, b = R[files[i]][task], R[files[j]][task]
                d = [(y - x) / a["n"] for x, y in zip(a["per_sampled"], b["per_sampled"])]
                lo, hi = boot(d)
                print(f"  {names[j]} - {names[i]}: {sum(d) / len(d):+.2%}  (95% CI {lo:+.2%} to {hi:+.2%})")


if __name__ == "__main__":
    main()
