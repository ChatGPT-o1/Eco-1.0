import json, math, random, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
sys.path.insert(0, "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad")

SP = "/tmp/claude-0/-home-user-Eco-1-0/64de4ff5-5b49-51e4-8f5a-aac7f7879924/scratchpad"
M = f"{SP}/Qwen3.5-2B"
OUT = f"{SP}/lora_honesty"
EPOCHS, ACC, LR, MAXLEN, SEED = 2, 8, 2e-4, 1024, 0
torch.set_num_threads(4)
random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(M)
HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

def encode(messages, finished=True):
    pieces = []
    for m in messages:
        if m["role"] == "user":
            pieces.append((f"<|im_start|>user\n{m['content']}<|im_end|>\n", False))
        else:
            pieces.append((HEADER, False))
            pieces.append((m["content"], True))
            if finished:
                pieces.append(("<|im_end|>", True))
                pieces.append(("\n", False))
    ids, labels = [], []
    for text, train in pieces:
        t = tok(text, add_special_tokens=False).input_ids
        ids += t
        labels += t if train else [-100] * len(t)
    return ids[:MAXLEN], labels[:MAXLEN]

data = []
for l in open(f"{SP}/train_single.jsonl"):
    r = json.loads(l); data.append(r["messages"])
n_single = len(data)
gens = {}
for l in open(f"{SP}/base_gen.jsonl"):
    r = json.loads(l)
    if r["id"].startswith("insist-train") and r["generated"].strip():
        gens[r["id"]] = (r["generated"], r["finished"])
ctx = [json.loads(l) for l in open(f"{SP}/insist_ctx.jsonl")]
n_insist = 0
fins = [True] * n_single
for c in ctx:
    if c["id"] in gens:
        txt, fin = gens[c["id"]]
        data.append(c["messages"] + [{"role": "assistant", "content": txt}])
        fins.append(fin)
        n_insist += 1
print(f"examples: single={n_single} insist={n_insist} total={len(data)}", flush=True)

enc = [encode(m, f) for m, f in zip(data, fins)]
enc = [e for e in enc if any(x != -100 for x in e[1])]
lens = sorted(len(e[0]) for e in enc)
print(f"tokens: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}", flush=True)

model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                 "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
model.train()

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
total_steps = math.ceil(len(enc) * EPOCHS / ACC)
warm = 5
def lr_at(s):
    if s < warm:
        return LR * (s + 1) / warm
    prog = (s - warm) / max(1, total_steps - warm)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

step = 0; micro = 0; t0 = time.time(); run_loss = 0.0; run_n = 0
for ep in range(EPOCHS):
    order = list(range(len(enc))); random.shuffle(order)
    for idx in order:
        ids, labels = enc[idx]
        out = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels]), use_cache=False)
        (out.loss / ACC).backward()
        run_loss += out.loss.item(); run_n += 1
        micro += 1
        if micro % ACC == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0:
                print(f"ep {ep+1} step {step}/{total_steps} loss {run_loss/run_n:.4f} lr {lr_at(step):.2e} elapsed {time.time()-t0:.0f}s", flush=True)
                run_loss = 0.0; run_n = 0
model.save_pretrained(OUT)
print(f"SAVED adapter to {OUT} in {time.time()-t0:.0f}s", flush=True)
