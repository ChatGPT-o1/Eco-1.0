"""Side-by-side qualitative samples (not used for scoring). usage: demo.py OUT.md [ADAPTER]"""
import sys
import gen
from schema import error_turn, review_prompt

PROMPTS = [
    ("vibe", [{"role": "user", "content": "need a quick fn to dedupe a list but keep the order"}]),
    ("vibe", [{"role": "user", "content": "parse 'HH:MM' strings into minutes since midnight, reject garbage"}]),
    ("repair", [
        {"role": "user", "content": "write chunk(xs, n) that splits a list into pieces of size n"},
        {"role": "assistant", "content": "```python\ndef chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs) - n, n)]\n```"},
        {"role": "user", "content": error_turn("FAILED: assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n  got: [[1, 2], [3, 4]]")}]),
    ("review", [{"role": "user", "content": review_prompt(
        "Return the median of a non-empty list of numbers.",
        "def median(xs):\n    xs = sorted(xs)\n    return xs[len(xs) // 2]")}]),
    ("review", [{"role": "user", "content": review_prompt(
        "Return True if s reads the same forwards and backwards.",
        "def is_pal(s):\n    return s == s[::-1]")}]),
]

out_path, adapters = sys.argv[1], [None] + sys.argv[2:3]
blocks = {p: [] for p in range(len(PROMPTS))}
for ad in adapters:
    model, tok = gen.load(ad)
    res = gen.generate(model, tok, [{"messages": m} for _, m in PROMPTS], max_new=700, batch=8)
    for i, (text, _) in enumerate(res):
        blocks[i].append((ad or "base", text))
with open(out_path, "w") as f:
    for i, (kind, msgs) in enumerate(PROMPTS):
        f.write(f"## {i + 1}. {kind}\n\n**Prompt (last turn):**\n\n```\n{msgs[-1]['content']}\n```\n\n")
        for name, text in blocks[i]:
            f.write(f"**{'tuned' if name != 'base' else 'base'}:**\n\n````\n{text}\n````\n\n")
print("wrote", out_path)
