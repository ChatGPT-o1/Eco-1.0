# model-surgery

Shrink a Hugging Face causal LM for one domain (default: coding) using only the steps that **held up
under testing**, with a HumanEval gate after every change so a broken model is never accepted.

| Step | Script | Effect | Cost to quality |
|---|---|---|---|
| 0 | `fetch_data.py` | Download + split a code dataset (counting vs held-out) | — |
| 1 | `textonly.py` | Drop vision tower and unused MTP/NextN head | None |
| 2 | `count_tokens.py` | Count token use on your domain data | — |
| 3 | `prune_vocab.py` | Keep only domain tokens (within an inflation budget), slice embeddings | Identical on text that uses only kept tokens; text with a dropped token gets split into unfamiliar pieces, so bits/byte rises a few % (SmolLM2-135M at a 1.5% budget: code +3.9%, chat +7%, HumanEval unchanged). Lower `--max-inflation` to cut less |
| 4 | `verify.py` | Tokenization identical, logits identical, generation works | Check only |
| 5 | `evaluate.py` | Bits/byte on code/chat/general + HumanEval pass@1 (PyTorch or GGUF) | Check only |
| 6 | `to_gguf.py` | Convert to GGUF and quantize (Q6_K default) | ~0.2% bits/byte at Q6_K |
| all | `pipeline.py` | Steps 1–6 with gates; writes `REPORT.md` | — |

Measured results (4-core CPU, no GPU):

| Model | Before | After pipeline | HumanEval gate |
|---|---|---|---|
| Qwen3.5-2B (manual run of these steps) | 2.27B params, 4.5 GB | 1.50B params (vocab 248k → 64k); Q6_K est. ~1.2 GB | 4/5 spot check after vocab step |
| Qwen3.5-0.8B (`pipeline.py`; multimodal + MTP) | 853M params, 1.7 GB | 556M (vocab 248k → 56k), **0.46 GB** at Q6_K, 31 tok/s; code bits/byte +2.7%, chat +0.7%, general +30% | 5/10 before and after |
| SmolLM2-135M-Instruct (`pipeline.py`) | 134.5M, 0.27 GB | 116.4M (vocab 49k → 17.7k), **0.12 GB** at Q6_K, 208 tok/s | 1/10 before and after |

## Quick start (for an agent)

```bash
python -m venv venv && . venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=5" tokenizers safetensors huggingface_hub pyarrow numpy sentencepiece
# optional, for GGUF: build llama-cpp-python from its sdist (see requirements.txt)

python -I fetch_data.py --out data/                      # ~850 MB on disk
python -I pipeline.py --model Qwen/Qwen3.5-2B --data data/ --work runs/q2b \
    --llama-cpp src/llama_cpp_python-*/vendor/llama.cpp --quants Q6_K
cat runs/q2b/REPORT.md
```

`pipeline.py` resumes: each step's output under `--work` is reused, so re-run the same command after a
crash. Run scripts with `python -I` (isolated mode) because they read downloaded files.

Useful flags: `--gate-problems 20` (HumanEval problems per gate; 164 = full), `--max-pass-drop 1`
(allowed regressions vs the text-only baseline), `--max-inflation 0.015` (vocab budget),
`--skip-vocab`, `--quants Q6_K Q8_0 Q4_K_M`. Each script also runs standalone; see its `--help`.

## Which models work

- **Steps 1, 5, 6:** any model `AutoModelForCausalLM` can load and llama.cpp can convert.
- **Step 3 (vocab):** byte-level BPE `tokenizer.json` only — Qwen 2/2.5/3/3.5, Llama-3.x, GPT-2/NeoX,
  SmolLM, most recent models. SentencePiece-style tokenizers (Llama-2, Mistral v1, Gemma) are refused
  with a message; `pipeline.py` then continues without the vocab step.
- Gains from the vocab step are largest on **small models with big vocabularies**: embeddings were 22%
  of Qwen3.5-2B (34% after removing vision) but only ~15% of an 8B model and a few % of a 70B.

## What did NOT work (don't repeat without new ideas)

Tested on Qwen3.5-2B; numbers are bits/byte on held-out code (lower is better) and HumanEval spot checks.

| Attempt | Params | Code bits/byte | HumanEval (5) |
|---|---|---|---|
| Vocab-pruned baseline | 1.50B | 0.698 | 4/5 |
| Drop 2 layers + keep 87.5% MLP neurons, no healing | 1.28B | 0.747 | 2/5 |
| Drop 4 layers + keep 70% MLP neurons, LoRA distillation (131k tokens) | 1.04B | 0.715 | **0/5** |
| Qwen2.5-Coder-0.5B-Instruct (for reference) | 0.49B | — | 5/5 |

- **Neuron/layer pruning by activation magnitude** removed problem-solving ability even when bits/byte
  recovered to within 2.6% of the baseline. Neurons are polysemantic: "low-code" neurons also carry
  pieces of code reasoning, and small models have no redundancy.
- **Bits/byte is not a proxy for working code.** Most code tokens are easy (`self.`, `return`, brackets);
  a broken model still predicts them. Always gate on executed tests — `pipeline.py` does.
- **Distillation at temperature 2** made raw-code bits/byte worse; temperature 1 plus 0.5×CE helped
  bits/byte but not HumanEval. Real healing needs billions of tokens (GPU), not ~10⁵.
- At this size, a model *trained* for code (Qwen2.5-Coder) beats a general model cut down to code.
  Ideas not tried here: code-calibrated SliceGPT (rotate + slice), pruning scored by HumanEval impact,
  sparse-autoencoder feature ablation, much larger healing runs.

## Troubleshooting

- **llama.cpp: `tensor 'blk.N.attn_norm.weight' not found`** — the GGUF advertises an MTP block that was
  removed. `textonly.py` sets the MTP layer count to 0 and `to_gguf.py` adds `--no-mtp`.
- **converter: `BPE pre-tokenizer was not recognized`** after vocab pruning — the tokenizer hash changed
  but its rules didn't; `to_gguf.py --pre-tokenizer-from <unpruned dir>` reuses the original's name.
- **Slow PyTorch on CPU** — use `--dtype fp32` (default); bf16 was ~3× slower without bf16 hardware.
  Qwen3.5 linear-attention layers run reference PyTorch code on CPU; GGUF/llama.cpp is faster for generation.
- **Disk** — a 2B run needs roughly 3× the model size free (text-only copy, pruned copy, bf16 GGUF).
