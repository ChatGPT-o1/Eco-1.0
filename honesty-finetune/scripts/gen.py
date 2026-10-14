import json, sys, time, torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad")
from build_render import render

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=None)
ap.add_argument("--input", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_new", type=int, default=240)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--check", action="store_true")
args = ap.parse_args()

SP = "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad"
M = f"{SP}/Qwen3.5-2B"
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(M)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
eos_ids = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])
eos_ids.add(tok.convert_tokens_to_ids("<|im_end|>"))

def run(prompts):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                             repetition_penalty=1.05, pad_token_id=tok.pad_token_id,
                             eos_token_id=list(eos_ids))
    res = []
    L = enc["input_ids"].shape[1]
    for row in out:
        gen = row[L:].tolist()
        finished = any(t in eos_ids for t in gen)
        res.append((tok.decode(gen, skip_special_tokens=True).strip(), finished))
    return res

if args.check:
    ps = [render([{"role": "user", "content": "What is the capital of France?"}], gen=True),
          render([{"role": "user", "content": "Name one prime number above 50 and say why it is prime."}], gen=True)]
    single = [run([p])[0][0] for p in ps]
    batched = [r[0] for r in run(ps)]
    print("BATCH CHECK single==batched:", single == batched, flush=True)
    for a, b in zip(single, batched):
        print("  single:", repr(a[:90]), "\n  batch :", repr(b[:90]), flush=True)

rows = []
for path in args.input:
    rows += [json.loads(l) for l in open(path)]
print(f"{len(rows)} prompts", flush=True)
t0 = time.time()
with open(args.out, "w") as f:
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        texts = [render(r["messages"], gen=True) for r in chunk]
        for r, (txt, fin) in zip(chunk, run(texts)):
            r = dict(r); r["generated"] = txt; r["finished"] = fin
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        print(f"batch {i//args.batch+1}: {min(i+args.batch,len(rows))}/{len(rows)} in {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
