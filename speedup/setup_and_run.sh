#!/bin/bash
# setup_and_run.sh - run the 1K-parameter speed-up agent on your own machine (Mac M1/M2/M3, Linux or Windows WSL).
#   bash setup_and_run.sh [loop_seconds]        default 300
set -e
cd "$(dirname "$0")"
SECS=${1:-300}
python3 -m venv .venv && source .venv/bin/activate
pip install -q torch transformers pandas pyarrow numpy
export PRUNE_ROOT="$PWD/data_cache" OUT_DIR="$PWD/runs" DEVICE=cpu
M=$PRUNE_ROOT/model; D=$PRUNE_ROOT/data/wikitext-103-raw-v1; mkdir -p "$M" "$D"
for f in config.json generation_config.json merges.txt model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
  [ -s "$M/$f" ] || curl -sSfL -o "$M/$f" "https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/$f"; done
for f in test-00000-of-00001 validation-00000-of-00001 train-00000-of-00002; do
  [ -s "$D/$f.parquet" ] || curl -sSfL -o "$D/$f.parquet" "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/$f.parquet"; done
SEARCH_N=32 VAL_N=32 TEST_N=64 CALIB_N=8 SEQ=128 CAL_RUNS=3 CAL_TOKENS=32 LOOP_SECONDS=$SECS python -u agent.py | tee runs.log
echo "Done. Report: runs/report.json   best architecture: runs/best_arch.json   log: runs.log"
