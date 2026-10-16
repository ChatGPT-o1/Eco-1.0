"""Single source of truth for every format the model sees or emits.

The data builder, trainer, evaluator and repair harness all import from here, so
a prompt template, code extractor, review format or tool-call parser can never
drift between training and evaluation (the parser-regression trap).

If you run an external eval suite, align it by editing ONLY this file.
"""
import ast
import builtins
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("ECO_WORK", ROOT / "work"))
BASE_MODEL = os.environ.get("BASE_MODEL", str(WORK / "Qwen3.5-2B"))

# --------------------------------------------------------------------------
# Rendering: always through the model's own chat template, non-thinking mode.
# --------------------------------------------------------------------------
CHAT_KW = {"enable_thinking": False}


def render(tok, messages, tools=None, gen=False):
    """Render messages exactly as the Qwen3.5 template does (incl. tool system block)."""
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=gen, **CHAT_KW)


# --------------------------------------------------------------------------
# Task prompts
# --------------------------------------------------------------------------
def code_prompt(task, style="spec"):
    """User turn asking for code. task needs: kind, text/prompt, tests, entry_point."""
    if task["kind"] == "humaneval":
        return f"Complete this function.\n\n```python\n{task['prompt'].rstrip()}\n```"
    if style == "signature":
        return f"Complete this function.\n\n```python\n{task['stub'].rstrip()}\n```"
    if style == "vibe":
        return f"{task['vibe']}\n\nshould pass:\n```python\n{task['tests'][0]}\n```"
    return f"{task['text']}\n\nYour code should pass this test:\n```python\n{task['tests'][0]}\n```"


def error_turn(error_text):
    """User turn that reports a failed check back to the model."""
    return f"That fails:\n```\n{error_text.strip()}\n```"


DUPLICATE_NOTE = ("Your last fix is functionally identical to the code that already failed. "
                  "Do not reuse it. The failing logic is:\n{culprit}\nTake a different approach.")


REVIEW_INSTR = ("Review this code for bugs before I ship it. Look for edge cases it gets wrong. "
                "Name the most likely bug (or what you checked), then end with exactly one line: "
                "FAILURE_PROBABILITY: <0-100>%")


def review_prompt(task_text, code):
    return f"Task: {task_text.strip()}\n\n```python\n{code.strip()}\n```\n\n{REVIEW_INSTR}"


def task_text(task):
    if task["kind"] == "humaneval":
        return task["prompt"].strip()
    return f"{task['text']} (e.g. `{task['tests'][0].replace('assert ', '', 1)}`)"


# --------------------------------------------------------------------------
# Code extraction
# --------------------------------------------------------------------------
FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)```", re.S)
OPEN_FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*)$", re.S)


def _calls_local(node, local_defs):
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in local_defs
               for n in ast.walk(node))


def sanitize(src):
    """Keep what defines the solution (imports, defs, classes, module assignments); drop the model's own
    example usage / self-tests (top-level asserts, prints, loops, __main__ blocks), evalplus-style.
    Unparseable code and code with no def/class is returned unchanged."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # truncated output: fall back to the longest parseable prefix (at most 30% of lines dropped)
        lines = src.splitlines()
        for cut in range(1, max(2, int(len(lines) * 0.3))):
            try:
                tree = ast.parse("\n".join(lines[:-cut]))
                src = "\n".join(lines[:-cut])
                break
            except SyntaxError:
                continue
        else:
            return src
    defs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    if not defs:
        return src
    keep = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep.append(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and not _calls_local(n, defs):
            keep.append(n)
        elif isinstance(n, ast.Try) and all(isinstance(b, (ast.Import, ast.ImportFrom)) for b in n.body):
            keep.append(n)
    lines = src.splitlines()
    parts = []
    for n in keep:
        start = min([d.lineno for d in getattr(n, "decorator_list", [])] + [n.lineno])
        parts.append("\n".join(lines[start - 1:n.end_lineno]))
    out = []
    for p in parts:
        if out and not (p.startswith(("import ", "from ")) and out[-1].startswith(("import ", "from "))):
            out.append("\n" if p.startswith(("def ", "class ", "@", "async ")) else "")
        out.append(p)
    return "\n".join(out).strip("\n") + "\n"


def extract_code(text, entry_point=None):
    """Return the code the model meant as its answer.

    Takes the last fenced block that defines `entry_point`, prefixed by earlier
    blocks that define helpers/imports. Falls back to an unterminated fence, then
    to the raw text.
    """
    blocks = FENCE.findall(text)
    if not blocks:
        m = OPEN_FENCE.search(text)
        blocks = [m.group(1)] if m else [text]
    blocks = [b.strip("\n") for b in blocks]
    if entry_point:
        pat = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", re.M)
        idx = [i for i, b in enumerate(blocks) if pat.search(b)]
        if idx:
            main = idx[-1]
            helpers = [b for b in blocks[:main]
                       if re.search(r"^\s*(def |class |import |from )", b, re.M) and not pat.search(b)]
            return sanitize("\n\n".join(helpers + [blocks[main]]))
    return sanitize(blocks[-1])


def prose_outside_code(text):
    """(words before first fence, words after last fence)."""
    first = text.find("```")
    if first < 0:
        return len(text.split()), 0
    last = text.rfind("```")
    return len(text[:first].split()), len(text[last + 3:].split())


# --------------------------------------------------------------------------
# Repair format: FAULT -> PIVOT -> code
# --------------------------------------------------------------------------
FAULT_RE = re.compile(r"^FAULT:\s*(.+)$", re.M)
PIVOT_RE = re.compile(r"^PIVOT:\s*(.+)$", re.M)


def repair_target(fault, pivot, code):
    return f"FAULT: {fault.strip()}\nPIVOT: {pivot.strip()}\n\n```python\n{code.strip()}\n```"


def parse_repair(text):
    f, p = FAULT_RE.search(text), PIVOT_RE.search(text)
    return (f.group(1).strip() if f else None, p.group(1).strip() if p else None)


# --------------------------------------------------------------------------
# Self-review format
# --------------------------------------------------------------------------
PROB_RE = re.compile(r"FAILURE_PROBABILITY:\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*%")


def review_target(finding, prob):
    return f"{finding.strip()}\nFAILURE_PROBABILITY: {int(prob)}%"


def parse_probability(text):
    m = PROB_RE.findall(text)
    if not m:
        return None
    return max(0.0, min(100.0, float(m[-1]))) / 100.0


# --------------------------------------------------------------------------
# Tool calls: Qwen3.5 template uses the qwen3_coder XML format.
# --------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(value, spec):
    t = (spec or {}).get("type")
    if t in ("integer", "number", "boolean", "array", "object"):
        try:
            return json.loads(value)
        except Exception:
            if t == "boolean" and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
            return value
    return value


def parse_tool_calls(text, tools=None):
    """Parse every <tool_call> block. Returns list of {name, arguments} (args typed by schema)."""
    specs = {}
    for t in tools or []:
        fn = t.get("function", t)
        specs[fn["name"]] = fn.get("parameters", {}).get("properties", {})
    calls = []
    for name, body in TOOL_CALL_RE.findall(text):
        name = name.strip()
        args = {k.strip(): _coerce(v, specs.get(name, {}).get(k.strip()))
                for k, v in PARAM_RE.findall(body)}
        calls.append({"name": name, "arguments": args})
    return calls


def tool_message(name, arguments, thought=""):
    """Assistant message with a tool call, rendered by the chat template itself."""
    return {"role": "assistant", "content": thought,
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


# --------------------------------------------------------------------------
# Voice checks (used as data filters and as eval metrics)
# --------------------------------------------------------------------------
FILLER = re.compile(
    r"\b(certainly|absolutely|of course|i'?d be (happy|glad)|happy to help|great question|"
    r"i apologi[sz]e|apologies|sorry|my (mistake|bad)|oversight|you'?re (absolutely )?right|"
    r"as an ai|i hope this helps|let me know if|feel free to)\b", re.I)


def filler_hits(text):
    return [m.group(0) for m in FILLER.finditer(text)]


PLACEHOLDER = re.compile(r"\bTODO\b|\bFIXME\b|^\s*\.\.\.\s*$|^\s*pass\s*#", re.M)


# --------------------------------------------------------------------------
# Normalisation for the duplicate blocker
# --------------------------------------------------------------------------
_BUILTINS = set(dir(builtins))


class _Norm(ast.NodeTransformer):
    """Strip docstrings and alpha-rename local identifiers (keeps the entry point name)."""

    def __init__(self, keep):
        self.keep, self.map = set(keep), {}

    def _n(self, name):
        if name in self.keep or name in _BUILTINS or name.startswith("__"):
            return name
        return self.map.setdefault(name, f"v{len(self.map)}")

    def _strip_doc(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]

    def visit_FunctionDef(self, node):
        self._strip_doc(node)
        if node.name not in self.keep:
            node.name = self._n(node.name)
        node.returns = None
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_arg(self, node):
        node.arg = self._n(node.arg)
        node.annotation = None
        return node

    def visit_Name(self, node):
        node.id = self._n(node.id)
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        return ast.Assign(targets=[node.target], value=node.value or ast.Constant(None), lineno=node.lineno)


def normalize_code(code, entry_point=None):
    """Canonical form for equality checks; falls back to whitespace-squashed text."""
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant):
            tree.body = tree.body[1:]
        tree = _Norm([entry_point] if entry_point else []).visit(tree)
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except SyntaxError:
        return re.sub(r"\s+", " ", re.sub(r"#.*", "", code)).strip()


def is_duplicate(a, b, entry_point=None):
    """Functionally identical modulo names, docstrings, comments, annotations, formatting."""
    return normalize_code(a, entry_point) == normalize_code(b, entry_point)


def similarity(a, b, entry_point=None):
    import difflib
    na, nb = normalize_code(a, entry_point), normalize_code(b, entry_point)
    if na == nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()


def jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
