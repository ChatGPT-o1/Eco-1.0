"""Load task banks into one dict format.

train pool : MBPP train + validation + prompt splits (474 tasks)
held-out   : HumanEval (164 tasks) -- never used to build training data.
"""
import re
import pandas as pd
from schema import WORK

DS = WORK / "ds"


def _stub(code, text, test0, entry):
    """Signature + docstring stub derived from the MBPP reference (for 'signature' prompts)."""
    m = re.search(rf"^def\s+{re.escape(entry)}\s*\(([^)]*)\)\s*:", code, re.M)
    args = m.group(1).strip() if m else ""
    return f'def {entry}({args}):\n    """{text.strip()}\n\n    >>> {test0.replace("assert ", "", 1)}\n    """\n'


def mbpp():
    rows = []
    for split in ("train", "validation", "prompt"):
        df = pd.read_parquet(DS / f"mbpp/full/{split}-00000-of-00001.parquet")
        for r in df.itertuples():
            tests = list(r.test_list)
            m = re.match(r"assert\s+\(?\s*(?:not\s+)?\(?\s*(?:set\()?\s*([A-Za-z_]\w*)\s*\(", tests[0])
            if not m:
                continue
            entry = m.group(1)
            code = r.code.replace("\r\n", "\n").replace("\t", "    ")
            rows.append({"id": f"mbpp/{r.task_id}", "kind": "mbpp", "text": r.text.strip(),
                         "tests": tests, "setup": r.test_setup_code or "", "entry_point": entry,
                         "reference": code, "stub": _stub(code, r.text, tests[0], entry)})
    return rows


def humaneval():
    df = pd.read_parquet(DS / "humaneval/openai_humaneval/test-00000-of-00001.parquet")
    return [{"id": r.task_id, "kind": "humaneval", "prompt": r.prompt, "test": r.test,
             "entry_point": r.entry_point, "reference": r.prompt + r.canonical_solution}
            for r in df.itertuples()]
