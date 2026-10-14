import json, re, subprocess, os, difflib
base='/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad/'
rows=[json.loads(l) for l in open(base+'fix2/reasons.jsonl')]
print('lines',len(rows))
for r in rows:
    assert set(r)=={'messages','finished','src'} and r['finished'] is True and r['src']=='reasons'
    assert [m['role'] for m in r['messages']]==['user','assistant']
old=[]
for f in ['hard/prompts.jsonl','train_trap.jsonl','fix/eval.jsonl','fix/easy.jsonl','fix/traps.jsonl','fix2/count.jsonl','fix2/reverse.jsonl','fix2/translate.jsonl']:
    for l in open(base+f):
        d=json.loads(l); old.append(d['messages'][0]['content'] if 'messages' in d else (d.get('prompt') or d.get('content')))
ps=[r['messages'][0]['content'] for r in rows]
assert len(set(ps))==30
for p in ps:
    best=max(old,key=lambda o: difflib.SequenceMatcher(None,p,o).ratio())
    rt=difflib.SequenceMatcher(None,p,best).ratio()
    if rt>0.6: print('SIMILAR',rt,p,'|',best)
env=dict(os.environ); env['PATH']=base+'venv/bin:'+env['PATH']
for i,r in enumerate(rows):
    a=r['messages'][1]['content']
    # sentence count: split on .?! followed by space+capital, ignoring inside backticks
    stripped=re.sub(r'`[^`]*`','CMD',a)
    sents=re.split(r'(?<=[.?!])\s+(?=[A-Z])',stripped)
    print(i,len(sents),'|',r['messages'][0]['content'][:60])
    for cmd in re.findall(r'`([^`]*)`',a):
        if cmd.startswith(('python3','echo')):
            out=subprocess.run(['bash','-c',cmd],capture_output=True,text=True,env=env)
            print('   $',cmd[:90],'->',repr(out.stdout[:120]),out.stderr[:200],out.returncode)
        else: print('   (non-cmd code)',cmd)
