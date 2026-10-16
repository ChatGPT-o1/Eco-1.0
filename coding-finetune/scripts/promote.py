"""Promotion gate: a checkpoint is promoted only if it STRICTLY beats base on
first-pass coding, failure detection and repair, and does not regress tools,
duplicate rate or voice.

usage: promote.py base tuned   (reads results/<name>.json) -> exit 0 if promoted
"""
import json
import sys
from schema import ROOT

b, t = (json.load(open(ROOT / f"results/{n}.json")) for n in sys.argv[1:3])
checks = [
    ("first-pass coding (pass@1) strictly higher", t["coding"]["pass"] > b["coding"]["pass"],
     f'{b["coding"]["pass"]}/{b["coding"]["n"]} -> {t["coding"]["pass"]}/{t["coding"]["n"]}'),
    ("failure detection (acc@50 on balanced review set) strictly higher",
     t["review"]["acc_at_50"] > b["review"]["acc_at_50"], f'{b["review"]["acc_at_50"]:.3f} -> {t["review"]["acc_at_50"]:.3f}'),
    ("review AUROC not lower", t["review"]["auroc"] >= b["review"]["auroc"],
     f'{b["review"]["auroc"]:.3f} -> {t["review"]["auroc"]:.3f}'),
    ("no 0% on broken code", t["review"]["zero_on_buggy"] == 0, f'{t["review"]["zero_on_buggy"]} zero-probability calls'),
    ("repair pass strictly higher", t["repair"]["pass"] > b["repair"]["pass"],
     f'{b["repair"]["pass"]}/{b["repair"]["n"]} -> {t["repair"]["pass"]}/{t["repair"]["n"]}'),
    ("repeats of failed code not higher", t["repair"]["duplicate_of_failed"] <= b["repair"]["duplicate_of_failed"],
     f'{b["repair"]["duplicate_of_failed"]} -> {t["repair"]["duplicate_of_failed"]}'),
    ("tool actions not lower", t["tools"]["pass"] >= b["tools"]["pass"],
     f'{b["tools"]["pass"]}/{b["tools"]["n"]} -> {t["tools"]["pass"]}/{t["tools"]["n"]}'),
    ("tool format fully parseable", t["tools"]["format_ok"] == 1.0, f'{t["tools"]["format_ok"]:.2f}'),
    ("filler/apology rate not higher (coding+repair)",
     t["coding"]["filler_rate"] + t["repair"]["filler_rate"] <= b["coding"]["filler_rate"] + b["repair"]["filler_rate"],
     f'{b["coding"]["filler_rate"] + b["repair"]["filler_rate"]:.3f} -> {t["coding"]["filler_rate"] + t["repair"]["filler_rate"]:.3f}'),
]
ok = all(c[1] for c in checks)
lines = [f"# Gate: {sys.argv[2]} vs {sys.argv[1]} -> {'PROMOTE' if ok else 'REJECT'}", ""]
lines += [f"- [{'x' if c[1] else ' '}] {c[0]}: {c[2]}" for c in checks]
print("\n".join(lines))
open(ROOT / f"results/gate_{sys.argv[2]}.md", "w").write("\n".join(lines) + "\n")
sys.exit(0 if ok else 1)
