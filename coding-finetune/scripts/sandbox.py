"""Execute candidate code against tests in an isolated subprocess.

Asserts are instrumented so a failure reports the failing assert and the actual
value (`got:`), like a real test runner, instead of a bare AssertionError. That
error text is what the model sees in repair turns, at train and at eval time.
"""
import ast
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

TIMEOUT = 8

RUNNER = r'''
import sys, traceback, io, contextlib
class _Fail(Exception):
    pass
def _fail(src, got=None, has_got=False):
    msg = "FAILED: " + src
    if has_got:
        r = repr(got)
        msg += "\n  got: " + (r if len(r) < 300 else r[:300] + "...")
    raise _Fail(msg)
_sol = open(sys.argv[1]).read()
_tst = open(sys.argv[2]).read()
_pre = open(sys.argv[3]).read() if len(sys.argv) > 3 else ""
_ctx = [None]
g = {"__name__": "__main__", "_fail": _fail, "_ctx": _ctx}
_out = io.StringIO()
try:
    with contextlib.redirect_stdout(_out):
        if _pre:
            exec(compile(_pre, "prelude.py", "exec"), g)
        exec(compile(_sol, "solution.py", "exec"), g)
except SystemExit:
    pass
except BaseException as e:
    tb = traceback.extract_tb(e.__traceback__)
    tb = [f for f in tb if f.filename == "solution.py"]
    lines = ["Traceback (most recent call last):"]
    for f in tb[-3:]:
        lines.append(f'  File "solution.py", line {f.lineno}, in {f.name}\n    {f.line}')
    lines.append(f"{type(e).__name__}: {e}")
    print("\n".join(lines)); sys.exit(3)
try:
    with contextlib.redirect_stdout(_out):
        exec(compile(_tst, "test.py", "exec"), g)
except _Fail as e:
    print(str(e)); sys.exit(1)
except BaseException as e:
    tb = traceback.extract_tb(e.__traceback__)
    keep = [f for f in tb if f.filename == "solution.py"]
    lines = []
    if _ctx[0]:
        lines.append("While checking: " + _ctx[0])
    lines.append("Traceback (most recent call last):")
    for f in keep[-3:]:
        lines.append(f'  File "solution.py", line {f.lineno}, in {f.name}\n    {f.line}')
    lines.append(f"{type(e).__name__}: {e}")
    print("\n".join(lines)); sys.exit(2)
print("PASS")
'''


class _Instrument(ast.NodeTransformer):
    def __init__(self, display=None):
        self.display = display or {}  # name shown in failure text, e.g. candidate -> below_zero

    def visit_Assert(self, node):
        self.generic_visit(node)
        seg = ast.unparse(node.test)
        for a, b in self.display.items():
            seg = seg.replace(f"{a}(", f"{b}(")
        src = "assert " + seg
        mark = ast.Assign([ast.Subscript(ast.Name("_ctx", ast.Load()), ast.Constant(0), ast.Store())], ast.Constant(src))
        t = node.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], (ast.Eq, ast.Is)):
            # __l = left; __r = right; if not (__l == __r): _fail(src, __l, True)
            l = ast.Name("__l", ast.Store()); r = ast.Name("__r", ast.Store())
            cmp = ast.Compare(ast.Name("__l", ast.Load()), t.ops, [ast.Name("__r", ast.Load())])
            call = ast.Expr(ast.Call(ast.Name("_fail", ast.Load()),
                                     [ast.Constant(src), ast.Name("__l", ast.Load()), ast.Constant(True)], []))
            return [mark, ast.Assign([l], t.left), ast.Assign([r], t.comparators[0]),
                    ast.If(ast.UnaryOp(ast.Not(), cmp), [call], [])]
        call = ast.Expr(ast.Call(ast.Name("_fail", ast.Load()), [ast.Constant(src)], []))
        return [mark, ast.If(ast.UnaryOp(ast.Not(), t), [call], [])]


def instrument(test_src, display=None):
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return test_src
    tree = _Instrument(display).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def test_program(task):
    """Return the (instrumented) test source for a task."""
    if task["kind"] == "humaneval":
        src = task["test"] + f"\n\ncheck({task['entry_point']})\n"
        return instrument(src, {"candidate": task["entry_point"]})
    setup = task.get("setup", "") or ""
    return instrument(setup + "\n" + "\n".join(task["tests"]) + "\n")


def prelude(task):
    """HumanEval convention: the prompt (imports + helper defs) runs first; the answer's defs override
    the stub. Run as a separate file so solution.py line numbers match the model's code."""
    return task["prompt"] + "\n    pass\n" if task["kind"] == "humaneval" else ""


def run(task, code, timeout=TIMEOUT):
    """-> (passed: bool, feedback: str)."""
    with tempfile.TemporaryDirectory() as d:
        sp, tp, rp, pp = (os.path.join(d, n) for n in ("solution.py", "test.py", "runner.py", "prelude.py"))
        open(sp, "w").write(code)
        open(tp, "w").write(test_program(task))
        open(rp, "w").write(RUNNER)
        open(pp, "w").write(prelude(task))
        try:
            p = subprocess.run([sys.executable, "-I", rp, sp, tp, pp], capture_output=True, text=True,
                               timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: tests did not finish within {timeout}s (likely an infinite loop or exponential blow-up)"
        out = (p.stdout or "").strip()
        if p.returncode == 0 and out.endswith("PASS"):
            return True, "PASS"
        if not out:
            out = (p.stderr or "").strip().splitlines()[-1:] or ["process exited with no output"]
            out = "\n".join(out)
        return False, out[-1500:]


def run_many(pairs, workers=4):
    """pairs: list of (task, code) -> list of (passed, feedback)."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda tc: run(*tc), pairs))
