# Running llm2ncm on a free Kaggle GPU

`build_kernel.py` packs `llm2ncm/` and `model-surgery/` into a self-contained Kaggle **script kernel**
(GPU + internet on, private). The sources are embedded in the script as a base64 tarball, so it works with a
private repo and needs no dataset.

| File | Purpose |
|---|---|
| `runner_template.py` | The kernel script (edit this). |
| `build_kernel.py` | Embeds the sources and writes `kernels/{full,smoke}/` (script + `kernel-metadata.json`). |
| `kernels/full/` | `--n 50000 --comp-steps 5000 --gen-steps 5000 --humaneval 20` on `HuggingFaceTB/SmolLM2-135M-Instruct`. |
| `kernels/smoke/` | Tiny configuration (~10 min) to check the GPU path before spending hours on the full run. |

`kernels/*/run_llm2ncm.py` are **generated**; rebuild after changing `llm2ncm/`, `model-surgery/` or the template.

## What the kernel does

1. Fails fast if no CUDA GPU is attached.
2. Unpacks the sources, `pip install -r model-surgery/requirements.txt` plus `transformers==5.19.0 peft==0.21.2
   accelerate==1.15.0` (versions that ran on Kaggle's image before), and removes Kaggle's `torchao`
   (0.10 breaks peft's LoRA injection).
3. `model-surgery/fetch_data.py` → data, `model-surgery/textonly.py` → text-only copy of the model.
4. `llm2ncm/selftest.py` (CPU mechanics check, non-fatal, logged), then `llm2ncm/pipeline.py --device cuda ...`.
5. Asks the trained model three prompts (`chat_samples.md`).
6. Exports to `/kaggle/working`: `REPORT.md`, `eval.json`, `generator/`, `compressor.pt(.json)`, `chat_samples.md`,
   `run.log`, `summary.json`. The ~2 GB `data.pt` is deliberately left out. If a stage crashes, whatever finished
   is still exported along with `FAILED.txt`.

## Workflow

```bash
export KAGGLE_API_TOKEN=...                       # never commit or print it
pip install kaggle
python kaggle/llm2ncm/build_kernel.py             # --owner <kaggle-username> if not timcook123
kaggle kernels list --mine                        # check auth
kaggle kernels push -p kaggle/llm2ncm/kernels/smoke --accelerator NvidiaTeslaT4
kaggle kernels status <owner>/llm2ncm-smollm2-135m-smoke
kaggle kernels push -p kaggle/llm2ncm/kernels/full  --accelerator NvidiaTeslaT4
kaggle kernels output <owner>/llm2ncm-smollm2-135m -p out/
```

The log is only available through `kaggle kernels output` after the run finishes.

## Using the downloaded generator elsewhere

`generator/config.json` stores the **absolute path** of the compressor used during training
(`/tmp/work/compressor.pt` on Kaggle). Point it at the downloaded `compressor.pt` before loading:

```bash
python -c "import json;p='out/generator/config.json';c=json.load(open(p));c['compressor']='out/compressor.pt';json.dump(c,open(p,'w'),indent=2)"
python -I llm2ncm/ncm_generate.py --model HuggingFaceTB/SmolLM2-135M-Instruct --generator out/generator \
    --prompt "What is 2 + 2?" --refine 4
```
