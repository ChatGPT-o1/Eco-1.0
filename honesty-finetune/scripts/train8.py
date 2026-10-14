import json, math, random, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
SP = "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad"
M = f"{SP}/Qwen3.5-2B"
torch.set_num_threads(4); random.seed(0); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(M)
HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
def encode(messages, finished=True):
    pieces = []
    for m in messages:
        if m["role"] == "user":
            pieces.append((f"<|im_start|>user\n{m['content']}<|im_end|>\n", False))
        else:
            pieces.append((HEADER, False)); pieces.append((m["content"], True))
            if finished:
                pieces.append(("<|im_end|>", True)); pieces.append(("\n", False))
    ids, labels = [], []
    for text, train in pieces:
        t = tok(text, add_special_tokens=False).input_ids
        ids += t; labels += t if train else [-100] * len(t)
    return ids[:1024], labels[:1024]
rows = [json.loads(l) for l in open(f"{SP}/train_v8.jsonl")]
enc = [encode(r["messages"], r.get("finished", True)) for r in rows]
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, f"{SP}/lora_honesty_v7", is_trainable=True)
model.train()
params = [p for p in model.parameters() if p.requires_grad]
LR, ACC = 5e-5, 4
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
order = list(range(len(enc))); random.shuffle(order)
t0 = time.time(); step = 0; micro = 0; run = 0.0; n = 0
for idx in order:
    ids, labels = enc[idx]
    out = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels]), use_cache=False)
    (out.loss / ACC).backward(); run += out.loss.item(); n += 1; micro += 1
    if micro % ACC == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups: g["lr"] = LR * min(1.0, (step + 1) / 3)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 3 == 0:
            print(f"step {step} loss {run/n:.4f} elapsed {time.time()-t0:.0f}s", flush=True); run = 0.0; n = 0
model.save_pretrained(f"{SP}/lora_honesty_v8")
print(f"SAVED v5 in {time.time()-t0:.0f}s", flush=True)
