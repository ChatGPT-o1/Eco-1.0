"""build.py - packs speedup/*.py into one Kaggle script.
    python kaggle/build.py <kernel_dir> <entry.py> [KEY=VALUE ...]      e.g.  python kaggle/build.py kaggle agent.py LOOP_SECONDS=300"""
import os, sys
src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kdir, entry = os.path.join(src, sys.argv[1]), sys.argv[2]; env = dict(a.split('=', 1) for a in sys.argv[3:])
mods = ['judge.py', 'smol.py', 'fastgen.py', 'quant.py', 'fused.py', 'agent.py']
files = {n: open(os.path.join(src, n)).read() for n in dict.fromkeys(mods + [entry])}
env = {'PRUNE_ROOT': '/tmp/prune', 'OUT_DIR': '/kaggle/working/runs', 'DEVICE': 'cpu', **env}
head = f'''# Auto-built by kaggle/build.py from speedup/ (entry: {entry})
import os, subprocess, sys
ROOT = '/tmp/prune'; os.makedirs(ROOT + '/data', exist_ok=True)
os.environ.update({env!r})
try:
    import cupy
except Exception:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda12x'], check=True)
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('HuggingFaceTB/SmolLM-135M', local_dir=ROOT + '/model', allow_patterns=['*.json', '*.safetensors', '*.txt'])
for f in ['test-00000-of-00001', 'validation-00000-of-00001', 'train-00000-of-00002']:
    hf_hub_download('Salesforce/wikitext', f'wikitext-103-raw-v1/{{f}}.parquet', repo_type='dataset', local_dir=ROOT + '/data')
'''
tail = f'''
os.makedirs('/kaggle/working/code', exist_ok=True)
for n, t in FILES.items(): open('/kaggle/working/code/' + n, 'w').write(t)
os.chdir('/kaggle/working/code')
subprocess.run([sys.executable, {entry!r}], check=True)
'''
out = os.path.join(kdir, 'run.py')
open(out, 'w').write(head + 'FILES = ' + repr(files) + '\n' + tail)
compile(open(out).read(), out, 'exec'); print('built', out, env)
