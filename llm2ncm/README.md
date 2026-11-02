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

**Measured** (SmolLM2-135M-Instruct, 50,000 windows, 5,000 compressor + 5,000 generator steps, one Kaggle T4, 47 min
end to end; files in `kaggle/llm2ncm/results/full-smollm2-135m/`, run from commit `fe2780f`): the NCM is 2.6x faster
than the original in the same PyTorch runtime (80.7 vs 31.3 tokens/s) but far worse. Judge 4.77 bits/token vs 0.55 for
the original (real human text scores 2.83), 6% token agreement, HumanEval 0/20 vs 2/20, distinct-2 0.23 vs 0.69. The
compressor recovers only 20% of tokens from its codes. Answers keep the topic words and the shape of a reply
("Here's a Python function: ``` def add ...") but the details are garbled. More refine passes did not help
(4.65 -> 4.77 bits from refine 1 to 4). Treat it as a research artifact, not a usable model.

**v2** (commits `dff8d51`..`03ca320`; files in `kaggle/llm2ncm/results/v2-smollm2-135m/`): trained on the original model's own
answers to chat prompts (cut where the NCM writes them), per-slot embeddings, compressor LR decay; two variants side by side.
Modest gains only: less repetition (distinct-2 0.23 -> 0.38/0.44) and a much better compressor for the wide variant (4+16 code slots:
43% of tokens recovered from codes vs 20%), but agreement with the original is 8% vs 6% (within the noise of 16 prompts), the judge
score is not better (4.9-5.2 vs 4.8 bits/token) and HumanEval stays 0/20. Answers now open correctly ("The capital of France is...",
"Here's a Python function that adds two numbers") and then degrade into repeated words. The limiting step is predicting the codes
from the prompt (level-2 code loss ~4.5 nats vs 6.9 for guessing), not turning codes into words.

**v3** (commits `3c9af7e`..`21c8950`; files in `kaggle/llm2ncm/results/v3-smollm2-135m/`): FSQ codes, a predictability term in compressor
training, an autoregressive first code level, and soft distillation, on the v2 data. No breakthrough. At refine=4 (match with the
original / judge bits per token / distinct-2): v3 9.7% / 5.33 / 0.35, v3-wide (4+16 slots) 7.8% / 4.43 / 0.37, against v2 8.4% / 5.24 /
0.38 and v2-wide 8.3% / 4.92 / 0.44. The best judge score so far (4.43, v3-wide) and the best France answer, but agreement did not move
beyond the noise of 16 prompts and HumanEval stays 0/20. The codes are measurably more predictable (the context model's cost is 1.31
nats per scalar vs 1.73 for guessing) and FSQ uses 49-96% of its codes (VQ: 33-84%), but the compressor gives up reconstruction for it
(2-level token recovery 20.4% / 32.9% vs 22.5% / 43.1%), and the generator's code loss did not improve. Passes per block rise to 8-10
because of the one-slot-at-a-time first level (about 2.2x / 1.8x faster than the original here, v2: 2.5x).
