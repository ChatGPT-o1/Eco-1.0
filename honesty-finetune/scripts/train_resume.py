import json, os, random, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
SP = "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad"
DATA, START, OUT = f"{SP}/train_chat200.jsonl", f"{SP}/lora_honesty_v7", f"{SP}/lora_chat_v1"
PROG = f"{OUT}_progress.json"
torch.set_num_threads(4); random.seed(0)
tok = AutoTokenizer.from_pretrained(f"{SP}/Qwen3.5-2B")
H = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
def enc(msgs, fin=True):
    ids, lab = [], []
    for m in msgs:
        parts = [(f"<|im_start|>user\n{m['content']}<|im_end|>\n", False)] if m["role"] == "user" else \
                [(H, False), (m["content"], True)] + ([("<|im_end|>", True), ("\n", False)] if fin else [])
        for t, tr in parts:
            x = tok(t, add_special_tokens=False).input_ids; ids += x; lab += x if tr else [-100] * len(x)
    return ids[:1024], lab[:1024]
rows = [json.loads(l) for l in open(DATA)]
data = [enc(r["messages"], r.get("finished", True)) for r in rows]
order = list(range(len(data))); random.shuffle(order)
done = json.load(open(PROG))["done"] if os.path.exists(PROG) else 0
model = AutoModelForCausalLM.from_pretrained(f"{SP}/Qwen3.5-2B", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, OUT if done else START, is_trainable=True)
model.train()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-5)
ACC, t0 = 4, time.time()
print(f"resuming at example {done}/{len(order)}", flush=True)
for i in range(done, len(order)):
    ids, lab = data[order[i]]
    out = model(input_ids=torch.tensor([ids]), labels=torch.tensor([lab]), use_cache=False)
    (out.loss / ACC).backward()
    if (i + 1) % ACC == 0 or i + 1 == len(order):
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        step = (i + 1) // ACC
        print(f"step {step} ex {i+1}/{len(order)} loss {out.loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if step % 5 == 0 or i + 1 == len(order):
            model.save_pretrained(OUT); json.dump({"done": i + 1}, open(PROG, "w"))
print("SAVED final", flush=True)
