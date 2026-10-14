import json, os, sys, time, urllib.request, urllib.error
SP = "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad"
MODEL = "claude-fable-5-1"
key = os.environ.get("ANTHROPIC_API_KEY", "")
if not key:
    sys.exit("ANTHROPIC_API_KEY is not set in this session; nothing was sent.")

def ask(prompt):
    body = json.dumps({"model": MODEL, "max_tokens": 400,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
                                 headers={"content-type": "application/json", "anthropic-version": "2023-06-01",
                                          "x-api-key": key})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
            return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"), data.get("stop_reason")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 529) and attempt < 2:
                time.sleep(5 * (attempt + 1)); continue
            raise

rows = []
for name in ["eval_prompts.jsonl", "insist_eval_ext.jsonl"]:
    rows += [json.loads(l) for l in open(f"{SP}/{name}")]
out = open(f"{SP}/fable_gen.jsonl", "w")
for r in rows:
    text, stop = ask(r["messages"][-1]["content"])
    r = dict(r); r["generated"] = text.strip(); r["finished"] = stop != "max_tokens"
    out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
out.close()
print(f"wrote {len(rows)} Fable replies to fable_gen.jsonl")
