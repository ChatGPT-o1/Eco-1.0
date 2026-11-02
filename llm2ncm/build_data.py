#!/usr/bin/env python3
"""Stage 0: training windows for llm2ncm.

Each window = prefix (P tokens) + block (L tokens), cut from code files and coding chats (rendered with
the model's chat template; chat blocks start inside the assistant reply). For --distill-frac of the
windows the block is replaced by the original LLM's own greedy continuation (sequence-level
distillation: the NCM learns to imitate this model, not just the data). Then the frozen LLM encodes
prefix+block and the hidden states of the block positions are stored for the compressor.

  python -I build_data.py --model DIR --data msdata/ --out work/data.pt --n 6000
"""
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from ncm_io import load_llm, pick_device


def read_jsonl(path, limit):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dir with count_code.jsonl and count_chat.jsonl (model-surgery/fetch_data.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--prefix", type=int, default=64)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--chat-frac", type=float, default=0.3)
    ap.add_argument("--distill-frac", type=float, default=0.5)
    ap.add_argument("--enc-layer", type=int, default=-1, help="hidden-state layer used as the encoder output")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    P, L = a.prefix, a.block
    dev = pick_device(a.device)
    tok, lm = load_llm(a.model, a.threads, dev)
    lm.eval()
    t0 = time.time()

    windows, sources = [], []
    n_chat = int(a.n * a.chat_frac)
    n_code = a.n - n_chat
    docs = read_jsonl(os.path.join(a.data, "count_code.jsonl"), n_code * 4)
    rng.shuffle(docs)
    for i in range(0, len(docs), 512):
        if len(windows) >= n_code:
            break
        enc = tok([d["text"] for d in docs[i:i + 512]], add_special_tokens=False)["input_ids"]
        for ids in enc:
            if len(ids) >= P + L and len(windows) < n_code:
                s = rng.randint(0, len(ids) - P - L)
                windows.append(ids[s:s + P + L]); sources.append("code")
    rows = read_jsonl(os.path.join(a.data, "count_chat.jsonl"), n_chat * 4)
    rng.shuffle(rows)
    for r in rows:
        if len(windows) >= a.n:
            break
        user = [{"role": "user", "content": r["instruction"]}]
        head = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(user + [{"role": "assistant", "content": r["response"]}], tokenize=False)
        ids = tok(full, add_special_tokens=False)["input_ids"]
        resp = len(tok(head, add_special_tokens=False)["input_ids"])
        lo = max(0, resp - P)
        hi = len(ids) - P - L
        if hi >= lo:
            s = rng.randint(lo, hi)
            windows.append(ids[s:s + P + L]); sources.append("chat")
    x = torch.tensor(windows, dtype=torch.long)
    print(f"windows: {len(windows)} ({sources.count('code')} code, {sources.count('chat')} chat) {time.time() - t0:.0f}s", flush=True)

    distilled = torch.zeros(len(x), dtype=torch.bool)
    idx = torch.randperm(len(x))[: int(len(x) * a.distill_frac)]
    with torch.inference_mode():
        for i in range(0, len(idx), a.batch):
            b = idx[i:i + a.batch]
            out = lm.generate(x[b, :P].to(dev), attention_mask=torch.ones(len(b), P, dtype=torch.long, device=dev),
                              max_new_tokens=L, min_new_tokens=L, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
            x[b, P:] = out[:, P:P + L].cpu()
            distilled[b] = True
            if (i // a.batch) % 10 == 0:
                print(f"  distilled {min(i + a.batch, len(idx))}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)

        H = lm.get_input_embeddings().weight.shape[1]
        hidden = torch.empty(len(x), L, H, dtype=torch.float16)
        for i in range(0, len(x), a.batch):
            hs = lm(x[i:i + a.batch].to(dev), output_hidden_states=True).hidden_states[a.enc_layer]
            hidden[i:i + a.batch] = hs[:, P:].to(torch.float16).cpu()
            if (i // a.batch) % 20 == 0:
                print(f"  encoded {min(i + a.batch, len(x))}/{len(x)} ({time.time() - t0:.0f}s)", flush=True)

    torch.save({"prefix": x[:, :P].to(torch.int32), "block": x[:, P:].to(torch.int32), "hidden": hidden,
                "distilled": distilled, "source": sources,
                "cfg": {"model": a.model, "P": P, "L": L, "enc_layer": a.enc_layer, "hidden": H}}, a.out)
    print(f"saved {a.out}: {len(x)} windows, hidden {tuple(hidden.shape)}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
