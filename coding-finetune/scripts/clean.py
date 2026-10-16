"""Trim a passing solution to its minimal form (callers re-verify it still passes).

Keeps imports, defs, classes and module constants exactly as written (comments
inside functions survive); drops docstrings, example usage, prints, asserts and
__main__ blocks.
"""
import ast


def _drop_docstrings(src):
    tree = ast.parse(src)
    cut = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and len(node.body) > 1:
            b = node.body[0]
            if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str):
                cut.append((b.lineno, b.end_lineno))
    lines = src.splitlines()
    for a, b in sorted(cut, reverse=True):
        del lines[a - 1:b]
    return "\n".join(lines)


def clean_code(src):
    from schema import sanitize
    code = sanitize(src)
    try:
        code = _drop_docstrings(code)
    except SyntaxError:
        pass
    return "\n".join(l.rstrip() for l in code.splitlines()).strip() + "\n"
