import json, io, contextlib
items = []
def add(u, a): items.append((u, a))

# Letter counting (7)
for w, l, terse in [("carrot","r",0),("banana","a",0),("cheese","e",0),("balloon","l",0),("cookbook","o",0),("lesson","s",0),("letter","t",1)]:
    assert len(w) <= 10
    n = w.count(l)
    if terse:
        add(f"Just the number: how many {l}'s are in '{w}'?", str(n))
    else:
        add(f"How many {l}'s are in the word '{w}'?", f"There are {n} {l}'s in '{w}'.")

# Reversal (6)
for w, terse in [("planet",0),("lemon",0),("garden",0),("stream",0),("cat",0),("house",1)]:
    assert len(w) <= 6
    r = w[::-1]
    if terse:
        add(f"Reverse '{w}'. Just the answer.", r)
    else:
        add(f"Reverse the word '{w}'.", f"'{w}' reversed is '{r}'.")

# Spelling (5)
for w in ["friend","school","rhythm","answer","because"]:
    add(f"Spell '{w}' letter by letter.", f"{w}: " + "-".join(w.upper()))

# Arithmetic (11)
arith = [("What is 47 + 38?", 47+38, 0), ("What is 256 + 489?", 256+489, 0), ("What is 900 - 357?", 900-357, 0),
         ("What is 63 - 29?", 63-29, 0), ("What is 23 times 17?", 23*17, 0), ("What is 12 x 15?", 12*15, 0),
         ("What is 99 * 11?", 99*11, 0), ("What is 84 divided by 7?", 84//7, 0), ("What is 625 / 5?", 625//5, 0),
         ("Just the number: 45 + 67", 45+67, 1), ("36 * 24? Number only.", 36*24, 1)]
assert 84 % 7 == 0 and 625 % 5 == 0
for q, v, terse in arith:
    if terse:
        add(q, str(v))
    else:
        expr = q[len("What is "):-1]
        add(q, f"{expr} = {v}.")

# Percentages (5)
for p, n, terse in [(10,250,0),(25,80,0),(50,300,0),(20,450,0),(15,200,1)]:
    v = p*n/100
    assert v == int(v); v = int(v)
    if terse:
        add(f"{p}% of {n}, just the number.", str(v))
    else:
        add(f"What is {p}% of {n}?", f"{p}% of {n} is {v}.")

# Code outputs (8)
codes = [("print(len('apple'))",0),("print(3*4)",0),("print('ab' * 3)",0),("print(10 // 3)",0),
         ("print(10 % 3)",0),("print('hello'.upper())",0),("print([1, 2, 3][-1])",0),("print(7 + 8)",1)]
for c, terse in codes:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(c)
    out = buf.getvalue().strip()
    if terse:
        add(f"Output only: {c}", out)
    else:
        add(f"What does this Python code print?\n{c}", f"It prints {out}")

# Facts (9)
facts = [
 ("What is the capital of France?", "The capital of France is Paris."),
 ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
 ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
 ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
 ("At what temperature does water boil at sea level, in Celsius?", "Water boils at 100 °C at sea level."),
 ("How many planets are in our solar system?", "There are eight planets in our solar system."),
 ("One word: capital of Italy?", "Rome"),
 ("One word: largest planet in our solar system?", "Jupiter"),
 ("Chemical symbol for sodium? One word.", "Na"),
]
for q,a in facts: add(q,a)

# Definitions (5)
defs = [
 ("What is a noun?", "A noun is a word that names a person, place, thing, or idea, such as \"teacher,\" \"city,\" or \"happiness.\""),
 ("What is photosynthesis?", "Photosynthesis is the process plants use to turn sunlight, water, and carbon dioxide into sugar (glucose), releasing oxygen as a by-product."),
 ("Define \"synonym.\"", "A synonym is a word that means the same or nearly the same as another word, like \"big\" and \"large.\""),
 ("What is a prime number?", "A prime number is a whole number greater than 1 whose only divisors are 1 and itself, such as 2, 3, 5, and 7."),
 ("One word: opposite of 'hot'?", "Cold"),
]
for q,a in defs: add(q,a)

# Rewrites (4)
rw = [
 ("Rewrite in passive voice: \"The cat chased the mouse.\"", "The mouse was chased by the cat."),
 ("Fix the grammar: \"She don't like apples.\"", "She doesn't like apples."),
 ("Rewrite in the past tense: \"I walk to school every day.\"", "I walked to school every day."),
 ("Make this more concise: \"Due to the fact that it was raining, we stayed inside.\"", "Because it was raining, we stayed inside."),
]
for q,a in rw: add(q,a)

assert len(items) == 60, len(items)
lines = [json.dumps({"messages":[{"role":"user","content":u},{"role":"assistant","content":a}],"finished":True,"src":"easy"}, ensure_ascii=False) for u,a in items]
open('/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad/easy_draft.jsonl','w').write("\n".join(lines)+"\n")
print("\n".join(lines))
