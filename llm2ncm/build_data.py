#!/usr/bin/env python3
"""Stage 0: training windows for llm2ncm.

Each window = prefix (P tokens) + block (L tokens). Three kinds:
  code    cut from code files
  chat    cut from coding chats (rendered with the model's chat template; blocks start inside the assistant reply)
  answer  (--prompt-frac) the ORIGINAL model's own greedy answer to a chat prompt, cut into its first --answer-blocks
          blocks exactly where the NCM writes them at inference (block 1 starts right after the prompt, block 2 after
          32 answer tokens, ...). This is on-policy self-distillation: the NCM learns to imitate what the original
          model actually says, in the situation it is used in. Prompts shorter than P are left-padded and
          `prefix_mask` marks the real tokens (padding is masked out of attention everywhere).
For --distill-frac of the code/chat windows the block is replaced by the LLM's own greedy continuation (answer
windows are always the model's own output). Then the frozen LLM encodes prefix+block and the hidden states of the
block positions are stored for the compressor.

  python -I build_data.py --model DIR --data msdata/ --out work/data.pt --n 6000
  python -I build_data.py ... --prompt-frac 0.5 --distill-frac 1.0      # v2 recipe (run fetch_prompts.py first)
"""
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from ncm_io import end_ids, load_llm, pick_device


def read_jsonl(path, limit):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= limit:
                break
    return out


def answer_windows(tok, lm, dev, prompts, n_want, P, L, blocks, batch, t0):
    """The original model answers `prompts` (token id lists) greedily, in left-padded batches; each answer gives up
    to `blocks` windows. Returns (x (N, P+L), prefix_mask (N, P), prompts used)."""
    stops = sorted(end_ids(tok))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    xs, ms, used = [], [], 0
    for i in range(0, len(prompts), batch):
        if len(xs) >= n_want:
            break
        group = prompts[i:i + batch]
        used += len(group)
        n_max = max(len(p) for p in group)
        ids = torch.tensor([[pad] * (n_max - len(p)) + p for p in group])
        att = torch.tensor([[0] * (n_max - len(p)) + [1] * len(p) for p in group])
        out = lm.generate(ids.to(dev), attention_mask=att.to(dev), max_new_tokens=blocks * L, do_sample=False,
                          eos_token_id=stops, pad_token_id=pad)
        for p, r in zip(group, out[:, n_max:].tolist()):
            end = next((k for k, t in enumerate(r) if t in stops), None)
            fill = r[end] if end is not None else stops[0]
            r = r[:end + 1] if end is not None else r
            for b in range(blocks):
                if b * L >= len(r) or len(xs) >= n_want:
                    break
                pre = (p + r[:b * L])[-P:]
                xs.append([pad] * (P - len(pre)) + pre + (r[b * L:(b + 1) * L] + [fill] * L)[:L])
                ms.append([False] * (P - len(pre)) + [True] * len(pre))
        print(f"  answers: {used} prompts -> {len(xs)}/{n_want} windows ({time.time() - t0:.0f}s)", flush=True)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ms, dtype=torch.bool), used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dir with count_code.jsonl and count_chat.jsonl (model-surgery/fetch_data.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--prefix", type=int, default=64)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--chat-frac", type=float, default=0.3, help="share of the non-answer windows cut from coding chats")
    ap.add_argument("--distill-frac", type=float, default=0.5, help="share of code/chat windows whose block is the LLM's own continuation")
    ap.add_argument("--prompt-frac", type=float, default=0.0,
                    help="share of all windows taken from the original model's own answers to chat prompts "
                         "(needs prompts.jsonl from fetch_prompts.py; 0 = the original recipe)")
    ap.add_argument("--prompt-code-share", type=float, default=0.3,
                    help="share of those prompts that are coding instructions (count_chat.jsonl); the rest from prompts.jsonl")
    ap.add_argument("--answer-blocks", type=int, default=3, help="blocks of each answer used as windows")
    ap.add_argument("--max-prompt", type=int, default=256, help="skip prompts longer than this many tokens")
    ap.add_argument("--enc-layer", type=int, default=-1, help="hidden-state layer used as the encoder output")
    ap.add_argument("--soft-k", type=int, default=0,
                    help="also store the original model's top-k next-token distribution at every block position (soft "
                         "distillation targets, train_generator.py --soft-weight); 0 = off")
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

    n_ans = int(a.n * a.prompt_frac)
    n_rest = a.n - n_ans
    windows, sources = [], []
    n_chat = int(n_rest * a.chat_frac)
    n_code = n_rest - n_chat
    if n_code:
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
    if n_chat:
        rows = read_jsonl(os.path.join(a.data, "count_chat.jsonl"), n_chat * 4)
        rng.shuffle(rows)
        for r in rows:
            if len(windows) >= n_rest:
                break
            user = [{"role": "user", "content": r["instruction"]}]
            head = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            full = tok.apply_chat_template(user + [{"role": "assistant", "content": r["response"]}], tokenize=False, enable_thinking=False)
            ids = tok(full, add_special_tokens=False)["input_ids"]
            resp = len(tok(head, add_special_tokens=False)["input_ids"])
            lo = max(0, resp - P)
            hi = len(ids) - P - L
            if hi >= lo:
                s = rng.randint(lo, hi)
                windows.append(ids[s:s + P + L]); sources.append("chat")
    x = torch.tensor(windows, dtype=torch.long).view(-1, P + L)
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

        prefix_mask = torch.ones(len(x), P, dtype=torch.bool)
        if n_ans:
            gen_rows = read_jsonl(os.path.join(a.data, "prompts.jsonl"), n_ans * 2)
            code_rows = read_jsonl(os.path.join(a.data, "count_chat.jsonl"), n_ans * 2)
            rng.shuffle(code_rows)
            k_code = int(n_ans * a.prompt_code_share)
            msgs = [[{"role": "user", "content": r["instruction"]}] for r in code_rows[:k_code]] + [r["messages"] for r in gen_rows]
            rng.shuffle(msgs)
            prompts = []
            for m in msgs:
                ids = tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
                if len(ids) <= a.max_prompt:
                    prompts.append(ids)
            ax, am, used = answer_windows(tok, lm, dev, prompts, n_ans, P, L, a.answer_blocks, a.batch, t0)
            x = torch.cat([x, ax])
            prefix_mask = torch.cat([prefix_mask, am])
            distilled = torch.cat([distilled, torch.ones(len(ax), dtype=torch.bool)])
            sources += ["answer"] * len(ax)
            print(f"answer windows: {len(ax)} from {used} prompts ({int((~am).sum())} padded prefix positions)", flush=True)

        H = lm.get_input_embeddings().weight.shape[1]
        hidden = torch.empty(len(x), L, H, dtype=torch.float16)
        ctx = torch.empty(len(x), H, dtype=torch.float16)  # summary of the prefix: hidden state at its last token
        if a.soft_k:
            topk_ids = torch.empty(len(x), L, a.soft_k, dtype=torch.int32)
            topk_p = torch.empty(len(x), L, a.soft_k, dtype=torch.float16)
        for i in range(0, len(x), a.batch):
            mb = prefix_mask[i:i + a.batch]
            kw = {}
            if not bool(mb.all()):  # left padding: mask the pads (an all-ones mask is left out to keep the old numerics)
                kw["attention_mask"] = torch.cat([mb, torch.ones(len(mb), L, dtype=torch.bool)], 1).long().to(dev)
            # logits_to_keep: skip the vocabulary projection at positions we do not need (L+1 = the block's predictions)
            o = lm(x[i:i + a.batch].to(dev), output_hidden_states=True, logits_to_keep=(L + 1 if a.soft_k else 1), **kw)
            hs = o.hidden_states[a.enc_layer]
            hidden[i:i + a.batch] = hs[:, P:].to(torch.float16).cpu()
            ctx[i:i + a.batch] = hs[:, P - 1].to(torch.float16).cpu()
            if a.soft_k:  # logits at position P-1+j predict block token j
                tp, ti = torch.softmax(o.logits[:, :L].float(), -1).topk(a.soft_k, -1)
                topk_ids[i:i + a.batch] = ti.int().cpu()
                topk_p[i:i + a.batch] = (tp / tp.sum(-1, keepdim=True)).half().cpu()
            if (i // a.batch) % 20 == 0:
                print(f"  encoded {min(i + a.batch, len(x))}/{len(x)} ({time.time() - t0:.0f}s)", flush=True)

    torch.save({"prefix": x[:, :P].to(torch.int32), "block": x[:, P:].to(torch.int32), "hidden": hidden,
                "distilled": distilled, "source": sources, "prefix_mask": prefix_mask, "ctx": ctx,
                **({"topk_ids": topk_ids, "topk_p": topk_p} if a.soft_k else {}),
                "cfg": {"model": a.model, "P": P, "L": L, "enc_layer": a.enc_layer, "hidden": H,
                        "prompt_frac": a.prompt_frac, "answer_blocks": a.answer_blocks, "soft_k": a.soft_k}}, a.out)
    print(f"saved {a.out}: {len(x)} windows ({sources.count('answer')} answer), hidden {tuple(hidden.shape)}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
