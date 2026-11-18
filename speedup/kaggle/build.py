"""build.py - packs speedup/*.py into one Kaggle script (kaggle/run_speedup.py).   python kaggle/build.py [loop_seconds]"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__)); src = os.path.dirname(here)
secs = sys.argv[1] if len(sys.argv) > 1 else '300'
files = {n: open(os.path.join(src, n)).read() for n in ['judge.py', 'smol.py', 'fastgen.py', 'agent.py']}
head = f'''# Auto-built by kaggle/build.py from speedup/ : a ~1K-parameter network learns to edit SmolLM-135M so it generates faster.
import os, subprocess, sys
ROOT = '/tmp/prune'; os.makedirs(ROOT + '/data', exist_ok=True)
os.environ.update(PRUNE_ROOT=ROOT, OUT_DIR='/kaggle/working/runs', LOOP_SECONDS='{secs}', DEVICE='cpu')
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('HuggingFaceTB/SmolLM-135M', local_dir=ROOT + '/model', allow_patterns=['*.json', '*.safetensors', '*.txt'])
for f in ['test-00000-of-00001', 'validation-00000-of-00001', 'train-00000-of-00002']:
    hf_hub_download('Salesforce/wikitext', f'wikitext-103-raw-v1/{{f}}.parquet', repo_type='dataset', local_dir=ROOT + '/data')
'''
tail = '''
os.makedirs('/kaggle/working/code', exist_ok=True)
for n, t in FILES.items(): open('/kaggle/working/code/' + n, 'w').write(t)
os.chdir('/kaggle/working/code')
subprocess.run([sys.executable, 'agent.py'], check=True)
'''
open(os.path.join(here, 'run_speedup.py'), 'w').write(head + 'FILES = ' + repr(files) + '\n' + tail)
compile(open(os.path.join(here, 'run_speedup.py')).read(), 'run_speedup.py', 'exec'); print('built', secs, 's')
