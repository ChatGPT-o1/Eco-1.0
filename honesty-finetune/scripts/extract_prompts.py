import json, glob, sys
files = ['hard/prompts.jsonl','train_trap.jsonl'] + sorted(glob.glob('fix/*.jsonl')) + sorted(glob.glob('fix2/*.jsonl')) + sorted(glob.glob('hard/*.jsonl'))
files = [f for f in dict.fromkeys(files) if not f.endswith('eval2.jsonl')]
out = []
for f in files:
    for i, line in enumerate(open(f)):
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except Exception as e:
            print("BAD", f, i, e, file=sys.stderr); continue
        ps=[]
        if 'prompt' in o: ps.append(o['prompt'])
        for m in o.get('messages',[]):
            if m.get('role')=='user': ps.append(m['content'])
        for p in ps: out.append((f,p))
json.dump(out, open('all_prompts.json','w'))
print(len(out))
