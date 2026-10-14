import json, re
L=[json.loads(l) for l in open('/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad/fix/traps.jsonl')]
def q(i):
    u=L[i]['messages'][0]['content']
    return re.search(r'"(.*)"',u).group(1)
for i,ch in [(5,'o'),(6,'n'),(7,'i'),(8,'t'),(8,'e'),(9,'a'),(49,'e'),(50,'r'),(51,'l'),(52,'o')]:
    t=q(i)
    print(i,ch,'words',len(t.split()),'lower-only',t.count(ch),'all',t.lower().count(ch),'capitals',[w for w in t.split() if ch.upper() in w])
# per-word checks
for i,ch in [(49,'e'),(50,'r'),(51,'l'),(52,'o')]:
    t=q(i)
    words=[re.sub(r'[^A-Za-z]','',w) for w in t.split()]
    print(i,[(w,w.lower().count(ch)) for w in words if w.lower().count(ch)])
    a=L[i]['messages'][1]['content']
    claimed=re.findall(r"(\w+) \((\d+)\)",a)
    for w,n in claimed:
        if w.lower().count(ch)!=int(n): print('  MISMATCH',w,n,w.lower().count(ch))
    print('  claimed sum',sum(int(n) for w,n in claimed), 'claimed words', [w for w,n in claimed])
