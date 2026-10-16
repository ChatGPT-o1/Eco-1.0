"""Inject single, realistic bugs into passing code with an exact diagnosis.

Every mutant is a one-site text edit, so the repair target (the original code) is
a surgical fix of the broken draft. Callers must verify the mutant fails tests.
Each mutant carries:
  fault : one sentence naming the root cause, quoting the broken line
  pivot : one sentence naming the fix
  finding: same as fault, used for self-review targets
"""
import ast
import io
import random
import re
import tokenize

CMP = {"<=": "<", "<": "<=", ">=": ">", ">": ">=", "==": "!=", "!=": "=="}


def _line(src, lineno):
    return src.splitlines()[lineno - 1].strip()


def _replace(src, line, col, end_col, new, end_line=None):
    lines = src.splitlines(keepends=True)
    end_line = end_line or line
    if end_line != line:
        return None
    s = lines[line - 1]
    lines[line - 1] = s[:col] + new + s[end_col:]
    return "".join(lines)


def _pick(rng, options):
    return rng.choice(options)


def _cmp_text(old, new, nl, ol, rng):
    if old in ("<=", ">=") and new in ("<", ">"):
        return (_pick(rng, [f"`{nl}` uses a strict `{new}`, so the case where both sides are equal is dropped.",
                            f"The strict `{new}` in `{nl}` misses the boundary value where both sides are equal."]),
                f"Make the comparison inclusive: `{ol}`.")
    if old in ("<", ">") and new in ("<=", ">="):
        return (_pick(rng, [f"`{nl}` uses `{new}`, so equal values slip through when they should be excluded.",
                            f"`{new}` in `{nl}` lets the equal case through; the bound must be strict."]),
                f"Use the strict comparison: `{ol}`.")
    return (_pick(rng, [f"`{nl}` has the condition inverted, so the branch runs for exactly the wrong items.",
                        f"The `{new}` in `{nl}` is backwards; it selects the cases that should be skipped."]),
            f"Flip the test back to `{old}`: `{ol}`.")


def mutants(src, seed=0):
    """Yield dicts {code, op, fault, pivot} for single-site mutations of src."""
    rng = random.Random(seed)
    try:
        tree = ast.parse(src)
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (SyntaxError, tokenize.TokenError, IndentationError):
        return []
    out = []

    def add(code, op, fault, pivot):
        if code and code != src:
            try:
                ast.parse(code)
            except SyntaxError:
                return
            out.append({"code": code, "op": op, "fault": fault, "pivot": pivot})

    # comparison / boolean / division operators (token level)
    for t in toks:
        if t.type == tokenize.OP and t.string in CMP:
            new = CMP[t.string]
            code = _replace(src, t.start[0], t.start[1], t.end[1], new)
            if code:
                nl, ol = _line(code, t.start[0]), _line(src, t.start[0])
                f, p = _cmp_text(t.string, new, nl, ol, rng)
                add(code, "cmp", f, p)
        elif t.type == tokenize.NAME and t.string in ("and", "or"):
            new = "or" if t.string == "and" else "and"
            code = _replace(src, t.start[0], t.start[1], t.end[1], new)
            if code:
                nl, ol = _line(code, t.start[0]), _line(src, t.start[0])
                f = (f"`{nl}` uses `or`, so either condition alone passes when both are required."
                     if new == "or" else
                     f"`{nl}` uses `and`, so it demands both conditions when either one should pass.")
                add(code, "boolop", f, f"Use `{t.string}`: `{ol}`.")
        elif t.type == tokenize.OP and t.string == "//":
            code = _replace(src, t.start[0], t.start[1], t.end[1], "/")
            if code:
                nl, ol = _line(code, t.start[0]), _line(src, t.start[0])
                add(code, "div", f"`{nl}` uses true division, which returns a float where an integer is needed.",
                    f"Use floor division: `{ol}`.")

    for node in ast.walk(tree):
        # range() bounds
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" \
                and node.args and not node.keywords and len(node.args) <= 2:
            stop = node.args[-1]
            seg = ast.get_source_segment(src, stop)
            if seg is None or stop.lineno != stop.end_lineno:
                continue
            m = re.fullmatch(r"(.+?)\s*\+\s*1", seg)
            new = m.group(1) if m else f"{seg} - 1"
            code = _replace(src, stop.lineno, stop.col_offset, stop.end_col_offset, new)
            if code:
                nl, ol = _line(code, stop.lineno), _line(src, stop.lineno)
                add(code, "range_end",
                    _pick(rng, [f"`{nl}` stops one short, so the final iteration never runs.",
                                f"The range in `{nl}` excludes the last value, an off-by-one at the end."]),
                    f"Include the end of the range: `{ol}`.")
            if len(node.args) == 1 or (isinstance(node.args[0], ast.Constant) and node.args[0].value == 0):
                if len(node.args) == 1:
                    s0 = ast.get_source_segment(src, node.args[0])
                    if s0 and node.args[0].lineno == node.args[0].end_lineno:
                        code = _replace(src, node.args[0].lineno, node.args[0].col_offset,
                                        node.args[0].end_col_offset, f"1, {s0}")
                else:
                    a0 = node.args[0]
                    code = _replace(src, a0.lineno, a0.col_offset, a0.end_col_offset, "1")
                if code:
                    nl, ol = _line(code, node.lineno), _line(src, node.lineno)
                    add(code, "range_start", f"`{nl}` starts at 1, so the first element (index 0) is skipped.",
                        f"Start from 0: `{ol}`.")
        # min <-> max
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
            new = "max" if node.func.id == "min" else "min"
            f = node.func
            code = _replace(src, f.lineno, f.col_offset, f.end_col_offset, new)
            if code:
                nl, ol = _line(code, f.lineno), _line(src, f.lineno)
                want = "largest" if node.func.id == "max" else "smallest"
                got = "smallest" if want == "largest" else "largest"
                add(code, "minmax", f"`{nl}` picks the {got} value, but the task needs the {want}.",
                    f"Use `{node.func.id}`: `{ol}`.")
        # accumulator initial value
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Constant) and node.value.value in (0, 1) \
                and type(node.value.value) is int:
            name = node.targets[0].id
            v = node.value
            new = "1" if v.value == 0 else "0"
            code = _replace(src, v.lineno, v.col_offset, v.end_col_offset, new)
            if code:
                if v.value == 0:
                    f = f"`{name}` starts at 1 instead of 0, so the result is skewed from the first step."
                else:
                    uses_mul = re.search(rf"\b{name}\s*\*=|\b{name}\s*=\s*{name}\s*\*", src)
                    f = (f"`{name}` starts at 0, so every multiplication into it stays 0."
                         if uses_mul else f"`{name}` starts at 0 instead of 1, so the first step is miscounted.")
                add(code, "init", f, f"Initialise `{name}` to {v.value}.")
        # first/last index
        if isinstance(node, ast.Subscript) and isinstance(node.slice, (ast.Constant, ast.UnaryOp)):
            sl = node.slice
            seg = ast.get_source_segment(src, sl)
            if seg in ("0", "-1") and sl.lineno == sl.end_lineno:
                new = "-1" if seg == "0" else "0"
                code = _replace(src, sl.lineno, sl.col_offset, sl.end_col_offset, new)
                if code:
                    nl, ol = _line(code, sl.lineno), _line(src, sl.lineno)
                    a, b = ("last", "first") if seg == "0" else ("first", "last")
                    add(code, "index", f"`{nl}` reads the {a} element where the {b} one is needed.",
                        f"Index the {b} element: `{ol}`.")
        # dropped normalisation
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not node.args \
                and node.func.attr in ("lower", "strip", "upper") and node.lineno == node.end_lineno:
            base = ast.get_source_segment(src, node.func.value)
            if base:
                code = _replace(src, node.lineno, node.col_offset, node.end_col_offset, base)
                if code:
                    nl, ol = _line(code, node.lineno), _line(src, node.lineno)
                    what = {"lower": "case", "upper": "case", "strip": "surrounding whitespace"}[node.func.attr]
                    add(code, "normalize", f"`{nl}` no longer normalises {what}, so equivalent inputs compare unequal.",
                        f"Restore `.{node.func.attr}()`: `{ol}`.")
        # sort order
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)) \
                and getattr(node.func, "id", getattr(node.func, "attr", None)) in ("sorted", "sort") \
                and node.lineno == node.end_lineno:
            seg = ast.get_source_segment(src, node)
            if seg and "reverse=True" in seg:
                code = src.replace(seg, re.sub(r",\s*reverse=True", "", seg), 1)
                f2 = "ascending" ; want = "descending"
            elif seg and seg.endswith(")") and "reverse" not in seg:
                code = src.replace(seg, seg[:-1] + (", " if (node.args or node.keywords) else "") + "reverse=True)", 1)
                f2 = "descending"; want = "ascending"
            else:
                code = None
            if code:
                nl, ol = _line(code, node.lineno), _line(src, node.lineno)
                add(code, "sort", f"`{nl}` sorts {f2}, but the result needs {want} order.",
                    f"Fix the sort order: `{ol}`.")
        # dropped early-return guard
        if isinstance(node, (ast.FunctionDef,)) and len(node.body) > 1:
            for st in node.body[:3]:
                if isinstance(st, ast.If) and not st.orelse and len(st.body) == 1 \
                        and isinstance(st.body[0], ast.Return):
                    lines = src.splitlines(keepends=True)
                    guard = "".join(lines[st.lineno - 1:st.end_lineno])
                    code = "".join(lines[:st.lineno - 1] + lines[st.end_lineno:])
                    cond = ast.get_source_segment(src, st.test)
                    g1 = " ".join(guard.split())
                    add(code, "guard", f"There's no guard for `{cond}`, so that input falls through to the general path and breaks.",
                        f"Handle it up front: `{g1}`.")
                    break
    rng.shuffle(out)
    return out
