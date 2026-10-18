# Coding + agentic fine-tune for Qwen3.5-2B

Goal: make a local 2B model work like a calm senior engineer: decisive first-pass code with a
one-line framing, a crisp FAULT → PIVOT when something fails (never the same broken code
twice), calibrated self-review, and exact tool calls, without losing base coding ability.

Everything here runs on CPU (4 cores, bf16/AMX). Results are in `results/RESULTS.md`.

## Layout

| Path | What it is |
|---|---|
| `scripts/schema.py` | **Single source of truth** for every format: chat rendering (the model's own template, non-thinking), task prompts, error turns, the code extractor + sanitizer, FAULT/PIVOT repair format, `FAILURE_PROBABILITY: N%` review format, qwen3_coder XML tool-call parser, voice checks, duplicate normaliser. Data builder, trainer, eval and harness all import it. Point an external eval suite here. |
| `scripts/sandbox.py` | Runs code against tests in a subprocess. Asserts are instrumented so failures read `FAILED: assert f(x) == y / got: z`, and tracebacks show only `solution.py` lines. |
| `scripts/collect_base.py` | Base model's greedy drafts on the MBPP pool (training) and HumanEval (held-out). |
| `scripts/annotations.py` | Hand-written repairs of the base model's **own** failed MBPP drafts (one-line FAULT, PIVOT, surgical edits or rewrite), first-pass framing lines, cleaner first-pass code overrides, and skipped items with reasons. |
| `scripts/mutate.py` | Single-site bug injection (comparison, off-by-one, init, min/max, index, guard, sort, …) with an exact diagnosis of the injected bug. |
| `scripts/build_data.py` | Builds `data/train.jsonl` (35% first-pass / 45% repair / 20% review + tools). Every code target is executed before it is written. |
| `scripts/train.py` | LoRA SFT. Only the final assistant turn is trained; every example asserts that the decoded trained tokens equal the target text exactly. |
| `scripts/harness.py` | Never-stuck inference loop: check → one surgical repair pass → duplicate blocker + re-sample with a negative constraint. |
| `scripts/build_eval.py`, `scripts/evaluate.py` | Frozen held-out sets (HumanEval-derived) and the scorer. |
| `scripts/promote.py` | Promotion gate. |
| `data/` | `train.jsonl`, frozen eval sets, base drafts. |

## Reproduce

```bash
export ECO_WORK=/path/to/workdir            # holds Qwen3.5-2B/, ds/ (humaneval, mbpp), venv
python scripts/collect_base.py humaneval && python scripts/collect_base.py mbpp
python scripts/build_eval.py && python scripts/check_ann.py && python scripts/build_data.py
sh scripts/run_smoke.sh                      # 60 s train + mini eval, base vs smoke adapter
python scripts/train.py --data data/train.jsonl --out $ECO_WORK/lora_full --lr 5e-5 --epochs 1
python scripts/evaluate.py --name base
python scripts/evaluate.py --name tuned --adapter $ECO_WORK/lora_full
python scripts/promote.py base tuned
python scripts/harness.py --name tuned --adapter $ECO_WORK/lora_full
```
