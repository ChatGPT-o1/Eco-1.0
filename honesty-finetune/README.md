# Honesty fine-tune for Qwen3.5-2B

Fine-tuning experiments that teach a small local model to say when a task is beyond its capability, and to attempt it when the user insists.

## Contents
- `scripts/`: data builders, LoRA training (`train*.py`), batched generation (`gen.py`), Fable comparison runner (`fable_eval.py`, needs `ANTHROPIC_API_KEY`).
- `data/`: training and eval JSONL files, including the round-by-round fix sets (`fix/`, `fix2/`) and the trap prompts (`hard/`).
- `results/`: eval generations, judge outputs and `RESULTS.md` with every score.
- `adapters/lora_chat_v4/`: final LoRA adapter (60 MB). It applies to the base model below.

## Not included
- Base weights `Qwen/Qwen3.5-2B` (4.5 GB). Download from Hugging Face.
- Other adapters (60–105 MB each). They are intermediate versions; the results table shows how they scored.
- Python environment.

## Paths
The scripts use absolute paths from the session that produced them (`/tmp/claude-0/.../scratchpad`). Point them at your own copy of the base weights and this folder before running.

## Status
See `results/RESULTS.md`. Blind-judge scores are from a Claude judge and a small sample; treat them as directional, not final. The Fable 5.1 comparison has not been run because no API key was available.
