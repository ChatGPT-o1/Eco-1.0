# llm2ncm — convert a pretrained LLM into a Next-Concept Model (experimental)

[Next-Concept Modeling](https://arxiv.org/abs/2610.14162) (NCM, Stanford Hazy Research / Together AI,
Oct 2026) writes text coarse-to-fine: a compressor turns a block of tokens into levels of discrete codes
(a rough outline → more detail), and a transformer predicts **one whole level per forward pass**. The paper
trains both parts from scratch. Nobody has published a way to convert an existing LLM into one; this is an
experimental attempt that reuses the LLM as much as possible and **measures what survives**.

## How the conversion works

| Role of the original LLM | Where |
|---|---|
| **Encoder**: its frozen hidden states are compressed into code levels (multiscale residual VQ, VAR-style) | `train_compressor.py` |
| **Generator backbone**: LoRA-tuned to predict each level in parallel; the last pass writes real tokens through the LLM's own output layer | `train_generator.py` |
| **Teacher**: half the training targets are its own continuations (sequence-level distillation) | `build_data.py` |
| **Judge**: scores every output; the NCM is always reported next to the original | `ncm_eval.py` |

One block of L tokens (default 32) after the prompt is laid out as
`prompt | level-1 slots | level-2 slots | L token slots`. Each slot sees the prompt, earlier levels and its
own level in both directions (custom 4D attention mask). Generating a block takes one pass per code level,
then R mask-predict passes over the tokens (`--refine`, keeping the most confident tokens each time), then
one pass to cache the finished block. The prompt and earlier levels are KV-cached, and `selftest.py`
checks that cached generation is token-for-token identical to full recomputation.

Passes per 32 tokens with levels `2,8`: R=1 → 4, R=2 → 5, R=4 → 7, versus 32 for normal generation.

## Supported models

- **Full-attention models** (Llama, Qwen2/2.5/3, Mistral, Gemma, SmolLM, Phi, …): the LLM itself is the
  generator backbone.
- **Models with linear-attention / recurrent layers** (Qwen3.5, Mamba hybrids, LFM2, …) can't look both
  ways inside a level. They still act as encoder, embeddings, output layer and teacher, but the generator is
  a small fresh transformer (`--backbone small`, chosen automatically and stated in the report). Expect
  much lower quality in that mode.

## Quick start

```bash
pip install torch transformers peft pyarrow   # plus model-surgery's requirements for the data
python -I ../model-surgery/fetch_data.py --out msdata/          # code + coding-chat data (~850 MB)
python -I selftest.py HuggingFaceTB/SmolLM2-135M-Instruct      # mechanics checks, ~2 min
python -I pipeline.py --model HuggingFaceTB/SmolLM2-135M-Instruct --data msdata/ --work runs/smol-ncm
cat runs/smol-ncm/REPORT.md
python -I ncm_generate.py --model HuggingFaceTB/SmolLM2-135M-Instruct --generator runs/smol-ncm/generator \
    --prompt "Write a Python function that reverses a string." --refine 4
```

`pipeline.py` resumes from whatever is already in `--work`. Main knobs: `--block`, `--scales`,
`--codebook`, `--n` (training windows), `--distill-frac`, `--comp-steps`, `--gen-steps`, `--lora-r`, `--refine`.

## What to expect (read before running)

- **Quality is not retained at CPU scale.** The paper trains on billions of tokens on GPUs; a CPU run sees
  a few hundred thousand. The report measures the loss honestly: judge bits/token (the original model's
  surprise at the output), agreement with the original's output, repetition, and HumanEval.
- **Speed:** fewer passes, but each pass processes up to 32 slots. On a CPU a 32-slot pass costs ~3× a
  1-token pass, so measured gains are ~1.7–3× over the same model generating normally in PyTorch. An
  optimized engine such as llama.cpp still beats it. On a GPU the gain should approach the pass reduction.
- **Fixed blocks:** a 5-token answer still pays for a full block; output appears block by block.
- The NCM paper reports that names and numbers suffer in compression, which matters for code identifiers.
