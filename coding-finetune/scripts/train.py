"""Conservative LoRA SFT with strict loss masking.

Only the final assistant turn of each example is trained (content + <|im_end|>).
Everything else -- system/tool schema, user turns, broken first drafts, earlier
tool calls, tool outputs -- gets labels = -100. Every example is checked at
encode time: the decoded trained tokens must equal the target text exactly.

usage: train.py --data data/train.jsonl --out work/lora_x [--max_seconds 60]
"""
import argparse
import json
import math
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from schema import BASE_MODEL, jsonl, render

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--max_len", type=int, default=1536)
ap.add_argument("--max_seconds", type=float, default=0, help="wall-clock budget (smoke test); 0 = none")
ap.add_argument("--check_only", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.set_num_threads(4)
random.seed(args.seed)
torch.manual_seed(args.seed)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
IM_END = "<|im_end|>"


def encode(row):
    msgs, tools = row["messages"], row.get("tools")
    assert msgs[-1]["role"] == "assistant"
    full = render(tok, msgs, tools)
    prefix = render(tok, msgs[:-1], tools, gen=True)
    assert full.startswith(prefix), "template prefix mismatch"
    end = full.index(IM_END, len(prefix)) + len(IM_END)
    target = full[len(prefix):end]
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    labels = [t if (s >= len(prefix) and e <= end) else -100 for t, (s, e) in zip(ids, offs)]
    trained = tok.decode([t for t in labels if t != -100])
    assert trained == target, f"mask mismatch:\n{trained!r}\n!=\n{target!r}"
    # nothing before the final assistant turn may carry a label
    first = next(i for i, l in enumerate(labels) if l != -100)
    assert all(l == -100 for l in labels[:first]) and offs[first][0] >= len(prefix)
    return ids, labels


rows = jsonl(args.data)
enc, skipped = [], 0
for r in rows:
    ids, labels = encode(r)
    if len(ids) > args.max_len:
        skipped += 1
        continue
    enc.append((ids, labels, r.get("kind", "?")))
n_tr = sum(sum(l != -100 for l in e[1]) for e in enc)
n_all = sum(len(e[0]) for e in enc)
print(f"{len(enc)} examples ({skipped} over {args.max_len} tok skipped); "
      f"{n_tr} trained / {n_all} total tokens ({n_tr / n_all:.1%} unmasked)", flush=True)
if args.check_only:
    raise SystemExit(0)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "out_proj"]
model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                                         task_type="CAUSAL_LM", target_modules=TARGETS))
for p in model.parameters():  # fp32 adapter weights for stable small-lr updates
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()
model.train()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))

order = []
for ep in range(math.ceil(args.epochs)):
    o = list(range(len(enc)))
    random.shuffle(o)
    order += o
order = order[:int(len(enc) * args.epochs)]
total_steps = max(1, len(order) // args.accum)
warm = max(1, int(0.1 * total_steps))


def lr_at(step):
    if step < warm:
        return args.lr * (step + 1) / warm
    prog = (step - warm) / max(1, total_steps - warm)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


t0 = time.time()
step, run, nrun, stopped = 0, 0.0, 0, False
hist = []
for k, idx in enumerate(order):
    ids, labels, kind = enc[idx]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels]), use_cache=False)
    (out.loss / args.accum).backward()
    run += out.loss.item()
    nrun += 1
    if (k + 1) % args.accum == 0:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        hist.append({"step": step, "loss": run / nrun, "grad_norm": gn, "lr": lr_at(step - 1),
                     "elapsed": round(time.time() - t0)})
        print(f"step {step}/{total_steps} loss {run / nrun:.4f} gn {gn:.2f} lr {lr_at(step - 1):.2e} "
              f"{time.time() - t0:.0f}s", flush=True)
        run, nrun = 0.0, 0
    if (k + 1) % len(enc) == 0 and (k + 1) < len(order):  # epoch boundary checkpoint
        ep = (k + 1) // len(enc)
        model.save_pretrained(f"{args.out}_ep{ep}")
        print(f"saved epoch {ep} checkpoint", flush=True)
    if args.max_seconds and time.time() - t0 > args.max_seconds:
        stopped = True
        if nrun:  # apply the partial accumulation so the smoke adapter is non-trivial
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.step()
            step += 1
            hist.append({"step": step, "loss": run / nrun, "elapsed": round(time.time() - t0)})
        print(f"time budget hit after {k + 1} examples / {step} steps", flush=True)
        break

model.save_pretrained(args.out)
json.dump({"args": vars(args), "examples_seen": k + 1, "steps": step, "stopped_early": stopped,
           "seconds": round(time.time() - t0), "history": hist}, open(args.out + "/train_log.json", "w"), indent=1)
print(f"SAVED {args.out} after {time.time() - t0:.0f}s", flush=True)
