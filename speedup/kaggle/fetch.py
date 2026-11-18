"""fetch.py - waits for a Kaggle run, downloads its output and extracts the event lines.   python kaggle/fetch.py <out_dir> [kernel-slug]"""
import json, os, subprocess, sys, time
out = sys.argv[1]; K = sys.argv[2] if len(sys.argv) > 2 else 'timcook123/smollm-speedup-1k-agent'; time.sleep(20)
while True:
    s = subprocess.run(['kaggle', 'kernels', 'status', K], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    s = s[0] if s else ''
    if any(k in s.lower() for k in ('complete', 'error', 'cancel')): break
    time.sleep(20)
print(s); subprocess.run(['rm', '-rf', out]); os.makedirs(out)
subprocess.run(['kaggle', 'kernels', 'output', K, '-p', out], capture_output=True)
log = [f for f in os.listdir(out) if f.endswith('.log')]
if log:
    with open(os.path.join(out, 'events.jsonl'), 'w') as ev:
        for e in json.load(open(os.path.join(out, log[0]))):
            t = e.get('data', '')
            if t.startswith('{"event"'): ev.write(t if t.endswith('\n') else t + '\n')
            elif any(k in t for k in ('Traceback', 'Error', 'error:')) and 'deprecat' not in t: print('LOG:', t.rstrip()[:400])
print(sorted(os.listdir(out)), sorted(os.listdir(os.path.join(out, 'runs'))) if os.path.isdir(os.path.join(out, 'runs')) else 'no runs dir')
