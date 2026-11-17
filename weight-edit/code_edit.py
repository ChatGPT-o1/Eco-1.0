#!/usr/bin/env python3
"""code_edit.py - hand-written coding rules, written straight into SmolLM2-135M's weights. No training of any kind.

    python code_edit.py --out smol-coder.gguf

Nothing here is fitted to examples. RULES below is a list the editor wrote by hand: "right after token A, Python code
usually continues with token B". Each rule becomes one neuron in the FIRST layer, built only from the model's own
vocabulary table:
    gate and up rows  = the direction of token A in the layer's input  -> the neuron fires only on token A
    down column       = the direction that raises token B at the output -> its logit goes up by `strength`
The first layer sees the current token almost unmixed, and whatever it writes is carried to the output by the
residual stream. The neurons replace the layer's least-used ones (chosen from the weights alone).
Then: 4-bit re-encode, save a GGUF, and report HumanEval before/after (code is executed against the tests).

STATUS: written, never run. The two measured constants below come from earlier sessions with this model.
"""
import argparse, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, 'model-surgery'))
import edit_model as EM
from edit_model import Lab, log, reencode, bake, write_gguf

F = torch.nn.functional
torch.set_grad_enabled(False)
FINAL_RMS = 32.0        # measured: typical size of the final residual stream in this model
MAX_NEURONS = 256       # at most this many of layer 0's 1,536 neurons are replaced
GATE_AT = 5.0          # the gate's input when the neuron sees its token (SiLU is ~linear and ~5 there)

# (token A, [tokens B that should follow], strength in logits). Written by hand from knowledge of Python.
RULES = [
 ("```", ["python"], 6), ("python", ["\n"], 3), ("def", [" "], 2), ("):", ["\n"], 4), ("]:", ["\n"], 3),
 (" return", [" "], 2), ("return", [" "], 2), (" import", [" "], 3), ("import", [" "], 3), (" from", [" "], 1),
 (" for", [" i", " x", " item"], 2), (" in", [" range", " "], 2), (" range", ["("], 5), ("range", ["("], 5),
 (" len", ["("], 5), ("len", ["("], 5), (" print", ["("], 5), ("print", ["("], 5), (" sum", ["("], 4),
 (" sorted", ["("], 5), (" max", ["("], 4), (" min", ["("], 4), (" int", ["("], 3), (" str", ["("], 3),
 (" enumerate", ["("], 5), (" isinstance", ["("], 5), (" zip", ["("], 4), (" abs", ["("], 4),
 (" if", [" "], 1), (" elif", [" "], 3), (" else", [":"], 5), ("else", [":"], 5), (" while", [" "], 2),
 (" try", [":"], 5), (" except", [" Exception", ":"], 3), (" finally", [":"], 5), (" lambda", [" x"], 3),
 (" not", [" in", " "], 1), (" is", [" None", " not"], 2), (" ==", [" "], 2), (" !=", [" "], 2),
 (".append", ["("], 6), (".join", ["("], 6), (".split", ["("], 6), (".lower", ["()"], 5), (".upper", ["()"], 5),
 (".strip", ["()"], 5), (".items", ["()"], 5), (".keys", ["()"], 5), (".values", ["()"], 5), (".get", ["("], 5),
 (" True", ["\n"], 1), (" False", ["\n"], 1), (" None", ["\n"], 1), (" self", ["."], 4), (" class", [" "], 2),
 (" yield", [" "], 2), (" raise", [" ValueError", " "], 3), (" with", [" open"], 2), (" open", ["("], 5),
]

# ---- upgrade: a much larger hand-written rule book, generated from lists the editor wrote ----
BUILTINS = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format", "getattr", "hasattr", "input", "int",
            "isinstance", "iter", "len", "list", "map", "max", "min", "next", "open", "ord", "chr", "pow", "print", "range",
            "repr", "reversed", "round", "set", "sorted", "str", "sum", "super", "tuple", "type", "zip"]
CALL_METHODS = ["append", "extend", "insert", "remove", "pop", "index", "count", "sort", "reverse", "copy", "join", "split",
                "replace", "startswith", "endswith", "find", "format", "get", "update", "add", "setdefault", "encode", "decode"]
NOARG_METHODS = ["lower", "upper", "strip", "lstrip", "rstrip", "items", "keys", "values", "isdigit", "isalpha", "isupper",
                 "islower", "title", "capitalize", "clear", "splitlines", "isspace", "isalnum"]
MODULES = {"math": [".sqrt", ".floor", ".ceil", ".pi"], "re": [".sub", ".findall", ".match", ".search"], "os": [".path"],
           "sys": [".argv"], "random": [".randint", ".choice"], "json": [".loads", ".dumps"], "collections": [".Counter", ".defaultdict"],
           "itertools": [".permutations", ".combinations"], "typing": [" import"], "heapq": [".heappush", ".heappop"]}
TYPES = ["List", "Dict", "Tuple", "Optional", "Set", "Any"]
EXCEPTIONS = ["ValueError", "TypeError", "KeyError", "IndexError", "Exception", "ZeroDivisionError"]
for b in BUILTINS: RULES += [(" " + b, ["("], 4), (b, ["("], 3), ("(" + b, ["("], 4)]
for m in CALL_METHODS: RULES.append(("." + m, ["("], 6))
for m in NOARG_METHODS: RULES.append(("." + m, ["()"], 5))
for mod, nxt in MODULES.items(): RULES += [(" " + mod, nxt, 2), (mod, nxt, 2)]
for t in TYPES: RULES.append((" " + t, ["["], 5))
for e in EXCEPTIONS: RULES += [(" " + e, ["(", ":"], 3)]
RULES += [
 # structure
 (":\n", ["    "], 4), ("\n\n", ["def", "```", "    "], 1), ('"""', ["\n"], 1), (" ->", [" int", " str", " bool", " List", " float"], 3),
 (" __", ["name", "init"], 3), ("__", ["("], 1), (" @", ["staticmethod", "property"], 2), (" async", [" def"], 5), (" await", [" "], 2),
 (" global", [" "], 2), (" assert", [" "], 2), (" del", [" "], 2), (" pass", ["\n"], 4), (" break", ["\n"], 4), (" continue", ["\n"], 4),
 # operators
 (" +=", [" 1", " "], 3), (" -=", [" 1", " "], 3), (" *=", [" "], 2), (" //", [" 2", " "], 2), (" %", [" 2", " "], 2), (" **", [" 2", " "], 2),
 (" <=", [" "], 2), (" >=", [" "], 2), (" and", [" "], 1), (" or", [" "], 1), (" =", [" "], 2),
 # comprehension and iteration
 (" [", ["x"], 1), ("[::-", ["1"], 6), ("[-", ["1"], 4), (" key", ["="], 4), (" reverse", ["="], 5), ("=lambda", [" x"], 4),
 (" i", [" in"], 1), (" x", [" in"], 1), (" item", [" in"], 2), (" _", [" in"], 3),
 # values
 (" ==", [" 0", " "], 1), (" return", [" result", " True", " False", " None", " "], 1), (" result", [" ="], 2), (" count", [" ="], 1),
 # suppress chatty prose inside a code block: after the opening fence, do not start a sentence
 ("```python", ["\n"], 5),
]

def single(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None

def write_rules(lab):
    """Overwrite spare first-layer neurons with one neuron per rule. Uses only the model's weights."""
    P = 'model.layers.0.mlp.'; FP = lab.FP
    G, U, D = FP[P + 'gate_proj.weight'], FP[P + 'up_proj.weight'], FP[P + 'down_proj.weight']
    E = FP['model.embed_tokens.weight']; Ebar = E.mean(0)
    g_in = FP['model.layers.0.post_attention_layernorm.weight']; g_out = FP['model.norm.weight']
    unit = lambda v: v / v.norm()
    # least-used neurons, judged from the weights alone: small input rows and a small output column
    spare = (G.norm(dim=1) * U.norm(dim=1) * D.norm(dim=0)).argsort().tolist()
    used = skipped = 0; seen = set()
    for a, bs, strength in RULES:
        ta = single(lab.tok, a); tbs = [t for t in (single(lab.tok, b) for b in bs) if t is not None]
        if ta is None or not tbs or ta in seen or used >= MAX_NEURONS: skipped += 1; continue
        seen.add(ta)                                                       # one neuron per trigger token; the first rule wins
        x = g_in * E[ta] / torch.sqrt((E[ta] ** 2).mean() + 1e-5)          # what the layer sees on token A (to first order)
        d = unit(g_in * (E[ta] - Ebar)); proj = float(x @ d)               # the token's own direction, and how strongly it shows
        u = g_out * (E[tbs].mean(0) - Ebar)                                # direction that raises the B tokens at the output
        n = spare[used]; used += 1
        G[n] = d * (GATE_AT / proj)                                        # gate input = 5 on token A, near 0 elsewhere
        U[n] = d / proj                                                    # up input = 1 on token A
        act = float(F.silu(torch.tensor(GATE_AT)))                         # so the neuron outputs ~5 on token A
        D[:, n] = u * (strength * FINAL_RMS / float(u @ u) / act)          # ... which lifts B's logit by `strength`
    return used, skipped

def humaneval(lab, path, limit, max_new):
    import evaluate as EV                                                  # model-surgery/evaluate.py: sandboxed grader
    probs = EV.load_humaneval(path, limit)
    reps = lab.generate([lab.user(EV.HE_INSTR.format(prompt=p['prompt'])) for p in probs], max_new=max_new, micro=16)
    return sum(EV.run_program(EV.build_program(p['prompt'], p['entry_point'], EV.extract_code(c), p['test']))[0] for p, c in zip(probs, reps)), len(probs)

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

    res = {}
    if not a.no_report:
        log('before: HumanEval on the plain 4-bit model'); he = humaneval_file(a.cache)
        lab.set_plain_4bit(); res['before (plain 4-bit)'] = humaneval(lab, he, a.humaneval_limit, a.max_new) + (lab.nll(lab.dev_pairs),)

    used, skipped = write_rules(lab)                    # edits the full-precision weights in place
    log(f'1. wrote {used} hand-written rule neurons into layer 0 ({skipped} rules skipped: a token was not a single piece)')
    log('2. re-encode everything to 4-bit (error-compensated rounding)')
    codes = reencode(lab); lab.set_weights(codes, last_mlp=lab.last_fp())
    Q, last = bake(lab, lab.last_fp(), lab.calib_last_inputs()); codes = dict(codes); codes.update(last)
    template = a.template if os.path.exists(a.template) else EM.build_template(lab, a.llama_cpp, a.cache)
    bad = write_gguf(lab, codes, template, a.out, a.llama_cpp)
    log(f'3. wrote {a.out}; read-back check: ' + ('all tensors exact' if not bad else f'PROBLEMS {bad[:5]}'))
    if bad: sys.exit(1)
    if a.no_report: return
    lab.set_weights(codes); res['after'] = humaneval(lab, he, a.humaneval_limit, a.max_new) + (lab.nll(lab.dev_pairs),)
    print('\n' + ' ' * 22 + 'HumanEval pass@1   general quality (lower is better)')
    for k, (p, n, q) in res.items(): print(f'{k:22s}{p:4d}/{n:<4d} ({100 * p / n:4.1f}%)   {q:.4f}')

if __name__ == '__main__':
    main()
