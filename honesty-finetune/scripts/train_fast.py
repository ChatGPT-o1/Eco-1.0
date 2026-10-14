import json, math, random, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
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
enc.sort(key=lambda e: len(e[0]))
BATCH = 4
batches = [enc[i:i + BATCH] for i in range(0, len(enc), BATCH)]
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"],
                 layers_to_transform=list(range(12, 24)))
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
model.train()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
random.shuffle(batches)
t0 = time.time()
for step, b in enumerate(batches, 1):
    L = max(len(x[0]) for x in b)
    ids = torch.full((len(b), L), tok.pad_token_id or 0, dtype=torch.long)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    att = torch.zeros((len(b), L), dtype=torch.long)
    for i, (x, y) in enumerate(b):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
    out = model(input_ids=ids, attention_mask=att, labels=lab, use_cache=False)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    for g in opt.param_groups: g["lr"] = 2e-4 * min(1.0, step / 3)
    opt.step(); opt.zero_grad(set_to_none=True)
    print(f"step {step}/{len(batches)} loss {out.loss.item():.4f} elapsed {time.time()-t0:.0f}s", flush=True)
model.save_pretrained(f"{SP}/lora_fast_test")
print(f"DONE in {time.time()-t0:.0f}s", flush=True)
