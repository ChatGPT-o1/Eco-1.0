#!/bin/bash
# runs on the Vast instance at boot: fetch code and data, run the decoder test and a 5-minute agent run; output goes to the instance log
python -c "import torch; [torch.zeros(1, device=f'cuda:{i}') for i in range(torch.cuda.device_count())]; print('GPU_OK', torch.cuda.device_count(), torch.__version__)" || { echo "GPU_BAD"; echo "=== ALLDONE"; exit 1; }
cd /root && git clone -q --depth 1 -b claude/pensive-davinci-5x8tz8 https://github.com/ChatGPT-o1/Eco-1.0.git eco && cd eco/speedup
pip install -q cupy-cuda12x transformers pandas pyarrow 2>&1 | tail -2
export PRUNE_ROOT=/root/prune OUT_DIR=/root/runs TRANSFORMERS_VERBOSITY=error
M=/root/prune/model; D=/root/prune/data/wikitext-103-raw-v1; mkdir -p $M $D
for f in config.json generation_config.json merges.txt model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
  curl -sSfL -o $M/$f https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/$f || echo "DL_FAIL $f"; done
for f in test-00000-of-00001 validation-00000-of-00001 train-00000-of-00002; do
  curl -sSfL -o $D/$f.parquet https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/$f.parquet || echo "DL_FAIL $f"; done
ls -la $M $D | grep -v total
echo "=== FUSED"; TEST_N=128 python fused_test.py 2>&1 | grep -v -e "Loading weights" -e deprecat
echo "=== AGENT"; LOOP_SECONDS=300 python agent.py 2>&1 | grep -v -e "Loading weights" -e deprecat -e '"probe"'
echo "=== ALLDONE"
