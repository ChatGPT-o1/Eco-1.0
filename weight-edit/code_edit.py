#!/usr/bin/env python3
"""code_edit.py - write Python coding knowledge straight into SmolLM2-135M's weights. One command, no gradient training.

    python code_edit.py --out smol-coder.gguf

The editor (the model that wrote this file) supplies the knowledge: SOLUTIONS below holds small Python tasks with
reference solutions, each asked in several wordings. The script:
  1. Re-encodes  every weight to 4-bit with error-compensated rounding (shared with edit_model.py, cached).
  2. Reads       the last layer's neuron activity at every token of every reference solution.
  3. Writes      one closed-form update of the last layer's output matrix so that, at each of those moments, the next
                 token of the solution wins by a margin, while the layer's output on ordinary text stays the same.
                 The strength is chosen on a wording that was held back from the solve.
  4. Saves       a 4-bit GGUF.
  5. Reports     HumanEval pass@1 (code executed against the tests) before and after, plus general-text quality.

STATUS: written, never run. Expect to debug on first run.
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, 'model-surgery'))
import edit_model as EM
from edit_model import Lab, log, reencode, bake, write_gguf

F = torch.nn.functional
torch.set_grad_enabled(False)

ASK = ["Write a Python function that {t}.", "Write a function in Python that {t}.", "Can you write Python code that {t}?",
       "Python: implement a function that {t}."]          # the last wording is held back to choose the strength

# (what the function does, reference solution)
SOLUTIONS = [
 ("returns the sum of a list of numbers", "def sum_list(nums):\n    return sum(nums)"),
 ("returns the largest number in a list", "def max_of(nums):\n    return max(nums)"),
 ("returns the smallest number in a list", "def min_of(nums):\n    return min(nums)"),
 ("returns the average of a list of numbers", "def average(nums):\n    return sum(nums) / len(nums)"),
 ("reverses a string", "def reverse_string(s):\n    return s[::-1]"),
 ("checks whether a string is a palindrome", "def is_palindrome(s):\n    return s == s[::-1]"),
 ("counts the vowels in a string", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
 ("checks whether a number is even", "def is_even(n):\n    return n % 2 == 0"),
 ("checks whether a number is prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
 ("returns the factorial of a number", "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"),
 ("returns the n-th Fibonacci number", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"),
 ("returns the greatest common divisor of two numbers", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"),
 ("removes duplicates from a list while keeping the order", "def unique(items):\n    seen = set()\n    result = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result"),
 ("returns only the even numbers from a list", "def evens(nums):\n    return [x for x in nums if x % 2 == 0]"),
 ("squares every number in a list", "def squares(nums):\n    return [x * x for x in nums]"),
 ("sorts a list in descending order", "def sort_desc(items):\n    return sorted(items, reverse=True)"),
 ("counts how many times each word appears in a string", "def word_count(text):\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts"),
 ("returns the longest word in a string", "def longest_word(text):\n    return max(text.split(), key=len)"),
 ("checks whether two strings are anagrams", "def is_anagram(a, b):\n    return sorted(a) == sorted(b)"),
 ("flattens a list of lists", "def flatten(lists):\n    return [x for sub in lists for x in sub]"),
 ("returns the second largest number in a list", "def second_largest(nums):\n    return sorted(set(nums))[-2]"),
 ("capitalizes the first letter of every word in a string", "def title_case(text):\n    return ' '.join(w.capitalize() for w in text.split())"),
 ("returns the sum of the digits of a number", "def digit_sum(n):\n    return sum(int(d) for d in str(abs(n)))"),
 ("merges two dictionaries", "def merge(a, b):\n    result = dict(a)\n    result.update(b)\n    return result"),
 ("returns the common elements of two lists", "def common(a, b):\n    return [x for x in a if x in b]"),
 ("checks whether a list is sorted", "def is_sorted(items):\n    return all(items[i] <= items[i + 1] for i in range(len(items) - 1))"),
 ("converts Celsius to Fahrenheit", "def to_fahrenheit(c):\n    return c * 9 / 5 + 32"),
 ("returns the index of a value in a list or -1 if it is missing", "def find(items, value):\n    for i, x in enumerate(items):\n        if x == value:\n            return i\n    return -1"),
 ("counts the words in a string", "def count_words(text):\n    return len(text.split())"),
 ("returns the product of a list of numbers", "def product(nums):\n    result = 1\n    for x in nums:\n        result *= x\n    return result"),
 ("removes all spaces from a string", "def remove_spaces(s):\n    return s.replace(' ', '')"),
 ("returns the numbers from 1 to n", "def one_to_n(n):\n    return list(range(1, n + 1))"),
 ("checks whether a year is a leap year", "def is_leap(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)"),
 ("returns the absolute difference of two numbers", "def abs_diff(a, b):\n    return abs(a - b)"),
 ("does a binary search on a sorted list and returns the index or -1", "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"),
 ("returns FizzBuzz values from 1 to n as a list of strings", "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out"),
]

def reply_text(code): return "```python\n" + code + "\n```"

def build_rows(lab):
    """Teacher-forced sequences. One row per solution token: (sequence index, position whose NEXT token is the target, target, wording)."""
    seqs, rows = [], []
    for ti, (task, code) in enumerate(SOLUTIONS):
        for wi, w in enumerate(ASK):
            base = lab.ids(lab.user(w.format(t=task)))
            ans = lab.tok.encode(reply_text(code), add_special_tokens=False) + [lab.IM_END]
            si = len(seqs); seqs.append(base + ans)
            for j, t in enumerate(ans): rows.append((si, len(base) - 1 + j, t, wi))
    return seqs, rows

def solve(lab, X, H, tgt, wi, mats, xq, margin=3.0, rounds=4, lam_grid=(0.1, 0.3, 1.0, 3.0)):
    """Closed form: dW = R K^T (K K^T + lam * C_text)^-1 on the last layer's output matrix, where K = neuron activity at
    each solution token, R = the push each of those moments still needs for its next token to win by `margin`, and
    C_text = activity statistics on ordinary text (keeps everything else in place)."""
    BASE = lab.last_fp(); Hm = H - lab.mlp_out(X, BASE)
    K = lab.acts(X, mats).double(); Kc = lab.acts(xq, mats).double(); Ccal = Kc.T @ Kc
    E = lab.model.model.embed_tokens.weight.detach().double(); gN = lab.model.model.norm.weight.detach().double()
    D0 = mats['down_proj'].double(); H0 = (Hm + lab.mlp_out(X, mats)).double()

    def logits(dW, idx): return lab.logits_from((H0[idx] + K[idx] @ dW.T).float()).double()
    def needs(dW, idx):
        h = H0[idx] + K[idx] @ dW.T; lg = lab.logits_from(h.float()).double(); t = tgt[idx]
        lt = lg.gather(1, t[:, None]).squeeze(1); lg.scatter_(1, t[:, None], -1e9); lc, comp = lg.max(1)
        need = (margin - (lt - lc)).clamp(min=0)
        u = gN[None, :] * (E[t] - E[comp])
        return (need * lab.rms(h.float()).double() / (u * u).sum(1))[:, None] * u
    def fit(idx, lam):
        Kf = K[idx]; A = Kf.T @ Kf; A = A + lam * (torch.trace(A) / torch.trace(Ccal)) * Ccal
        Ainv = torch.linalg.inv(A + 1e-6 * torch.trace(A) / A.shape[0] * torch.eye(A.shape[0], dtype=A.dtype))
        dW = torch.zeros_like(D0); R = torch.zeros(len(idx), D0.shape[0], dtype=torch.float64)
        for _ in range(rounds):
            R = R + needs(dW, idx); dW = (R.T @ Kf) @ Ainv
        return dW
    def acc(dW, idx): return float((logits(dW, idx).argmax(1) == tgt[idx]).float().mean())
    shift = lambda dW: float((Kc @ dW.T).norm(dim=1).mean() / (Kc @ D0.T).norm(dim=1).mean())

    hold = len(ASK) - 1; tr = torch.nonzero(wi != hold).flatten(); va = torch.nonzero(wi == hold).flatten()
    zero = torch.zeros_like(D0)
    log(f'    next-token accuracy before: seen wordings {acc(zero, tr):.3f}, held-back wording {acc(zero, va):.3f}')
    best = None
    for lam in lam_grid:
        dW = fit(tr, lam); a = acc(dW, va)
        log(f'    strength {lam:<4}: seen {acc(dW, tr):.3f}, held-back {a:.3f}, change on ordinary text {shift(dW) * 100:.1f}%')
        if best is None or a > best[0] + 0.003: best = (a, lam)
    dW = fit(torch.arange(len(tgt)), best[1])
    log(f'    chosen strength {best[1]} (held-back accuracy {best[0]:.3f}); written with all wordings')
    return (D0 + dW).float()

def humaneval(lab, path, limit, max_new):
    import evaluate as EV                               # model-surgery/evaluate.py: sandboxed grader
    probs = EV.load_humaneval(path, limit)
    reps = lab.generate([lab.user(EV.HE_INSTR.format(prompt=p['prompt'])) for p in probs], max_new=max_new, micro=16)
    passed = sum(EV.run_program(EV.build_program(p['prompt'], p['entry_point'], EV.extract_code(c), p['test']))[0] for p, c in zip(probs, reps))
    return passed, len(probs)

def humaneval_file(cache):
    p = os.path.join(REPO, 'model-surgery', 'data', 'humaneval.parquet')
    if os.path.exists(p): return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download('openai/openai_humaneval', 'openai_humaneval/test-00000-of-00001.parquet', repo_type='dataset', local_dir=cache)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    dm = os.path.join(REPO, 'model-surgery', 'runs', 'SmolLM2-135M-Instruct')
    ap.add_argument('--model', default=dm if os.path.isdir(dm) else 'HuggingFaceTB/SmolLM2-135M-Instruct')
    ap.add_argument('--out', default=os.path.join(HERE, 'smol-coder.gguf'))
    ap.add_argument('--template', default=os.path.join(REPO, 'model-surgery', 'runs', 'gguf', 'SmolLM2-135M-Instruct-Q4_0.gguf'))
    ap.add_argument('--llama-cpp', default='/home/user/ggml-org/llama.cpp')
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    ap.add_argument('--humaneval-limit', type=int, default=164); ap.add_argument('--max-new', type=int, default=300)
    ap.add_argument('--no-report', action='store_true'); ap.add_argument('--threads', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); os.makedirs(a.cache, exist_ok=True)
    EM.LOGF = open(os.path.join(a.cache, 'code_edit.log'), 'a')
    log('loading', a.model); lab = Lab(a)
    c = lambda n: os.path.join(a.cache, n)

    if os.path.exists(c('reencoded.pt')): codes = torch.load(c('reencoded.pt')); log('1. re-encode: cached')
    else: log('1. re-encode'); codes = reencode(lab); torch.save(codes, c('reencoded.pt'))
    lab.set_weights(codes, last_mlp=lab.last_fp())
    if os.path.exists(c('calib_last.pt')): xq = torch.load(c('calib_last.pt'))
    else: xq = lab.calib_last_inputs(); torch.save(xq, c('calib_last.pt'))

    seqs, rows = build_rows(lab)
    log(f'2. read: {len(SOLUTIONS)} solutions x {len(ASK)} wordings = {len(rows)} solution tokens')
    pos = {}
    for si, p, _, _ in rows: pos.setdefault(si, []).append(p)
    Xs, Hs = lab.capture_last(seqs, positions=[pos[i] for i in range(len(seqs))])
    X = torch.cat(Xs); H = torch.cat(Hs)                # same order as rows (rows are grouped by sequence, positions ascending)
    tgt = torch.tensor([t for _, _, t, _ in rows]); wi = torch.tensor([w for _, _, _, w in rows])

    log('3. write: one closed-form update of the last layer')
    mats = lab.last_fp(); mats['down_proj'] = solve(lab, X, H, tgt, wi, mats, xq)

    log('4. save'); Q, last = bake(lab, mats, xq); codes = dict(codes); codes.update(last)
    torch.save({'codes': codes}, c('code_final_codes.pt'))
    template = a.template if os.path.exists(a.template) else EM.build_template(lab, a.llama_cpp, a.cache)
    bad = write_gguf(lab, codes, template, a.out, a.llama_cpp)
    log(f'    wrote {a.out}; read-back check: ' + ('all tensors exact' if not bad else f'PROBLEMS {bad[:5]}'))
    if bad: sys.exit(1)
    if a.no_report: return

    log('5. report: HumanEval (code is executed against the tests) and general-text quality')
    he = humaneval_file(a.cache); res = {}
    for name, setup in (('before (plain 4-bit)', lab.set_plain_4bit), ('after', lambda: lab.set_weights(codes))):
        setup(); p, n = humaneval(lab, he, a.humaneval_limit, a.max_new)
        res[name] = (p, n, lab.nll(lab.dev_pairs), lab.nll(lab.new_pairs))
        log(f'    {name}: HumanEval {p}/{n}')
    print('\n' + ' ' * 22 + 'HumanEval pass@1   quality dev   quality unseen (lower is better)')
    for k, (p, n, d, u) in res.items(): print(f'{k:22s}{p:4d}/{n:<4d} ({100 * p / n:4.1f}%)   {d:.4f}        {u:.4f}')

if __name__ == '__main__':
    main()
