# SmolLM-135M speed-up agent

A ~1,000-parameter neural network learns, by reinforcement, which parts of SmolLM-135M to remove
(attention or MLP sublayers, MLP neurons, key/value head groups, 16-bit weights) so it generates
text faster while perplexity stays within 10% of the original. No weights are retrained.

## Run it on your machine
    bash setup_and_run.sh          # installs into .venv, downloads the model (~540 MB) and text (~300 MB), runs 5 minutes
    bash setup_and_run.sh 900      # 15-minute learning loop
Needs Python 3.10+ and ~2 GB of disk. On an Apple Silicon Mac (M1/M2/M3) it runs on the GPU through PyTorch's MPS
backend, with 16-bit weights available as an edit; elsewhere it uses an NVIDIA GPU if present, else the CPU.
Force the CPU with `DEVICE=cpu bash setup_and_run.sh`.

Results: `runs/report.json` (speed, perplexity on unseen test text, sample text), `runs/best_arch.json`, `runs/episodes.jsonl`.

## Files
- `agent.py`   the 1K network, the episodes, scoring shortcuts, the save checks, the final report
- `smol.py`    exact evaluation of any edited architecture without rebuilding the model
- `fastgen.py` decoder used for speed timing (CUDA graphs on NVIDIA GPUs, plain loop on CPU)
- `judge.py`   data splits (search / validation / test / calibration), reference perplexity
- `quant.py`, `fused.py`  integer weight formats and the custom-kernel GPU decoder (NVIDIA only, needs CuPy)
- `test_exact.py` checks the fast evaluation against HuggingFace
