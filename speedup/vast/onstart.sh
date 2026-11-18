#!/bin/bash
# runs on the Vast instance at boot: fetch code, install, run the decoder test and a 5-minute agent run; all output goes to the instance log
cd /root && git clone -q --depth 1 -b claude/pensive-davinci-5x8tz8 https://github.com/ChatGPT-o1/Eco-1.0.git eco && cd eco/speedup
pip install -q cupy-cuda12x transformers pandas pyarrow huggingface_hub 2>&1 | tail -2
export PRUNE_ROOT=/root/prune OUT_DIR=/root/runs HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('HuggingFaceTB/SmolLM-135M', local_dir='/root/prune/model', allow_patterns=['*.json', '*.safetensors', '*.txt'])
for f in ['test-00000-of-00001', 'validation-00000-of-00001', 'train-00000-of-00002']:
    hf_hub_download('Salesforce/wikitext', f'wikitext-103-raw-v1/{f}.parquet', repo_type='dataset', local_dir='/root/prune/data')
PY
echo "=== FUSED"; TEST_N=128 python fused_test.py 2>&1 | grep -v -e "Loading weights" -e deprecat
echo "=== AGENT"; LOOP_SECONDS=300 python agent.py 2>&1 | grep -v -e "Loading weights" -e deprecat -e '"probe"'
echo "=== ALLDONE"
