import json, re, sys
d=json.load(open('all_prompts.json'))
text='\n'.join(p for f,p in d)
words=set(w.lower() for w in re.findall(r"[A-Za-z]+", text))
nums=set(n.replace(',','') for n in re.findall(r"\d[\d,]*\.?\d*", text))
nums=set(n.rstrip('.') for n in nums)
json.dump({'words':sorted(words),'nums':sorted(nums)}, open('vocab.json','w'))
if __name__=='__main__':
    for w in sys.argv[1:]:
        lw=w.lower()
        hit = lw in words if not lw.replace(',','').isdigit() else lw.replace(',','') in nums
        # also substring check in text for words
        sub = lw in text.lower()
        print(f"{w}: token={'USED' if hit else 'new'} substring={'USED' if sub else 'new'}")
