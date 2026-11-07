#!/usr/bin/env python3
"""Shared engine for plain RL (GRPO) and V8: fast sampling, test grading, problem pool, atomic checkpoints.

Sampling is a hand-written decode loop with a KV cache:
  * each prompt is prefilled once and its cache copied for the n samples;
  * a row stops at an end token, at max_new, or as soon as its text holds a closed ```python block (extract_code only ever
    reads the first closed block, so nothing after it can change the grade);
  * finished rows are dropped from the batch, so long answers do not keep short ones busy.
Grading is binary: 1 if the extracted function passes every MBPP test, else 0.
"""
import json, math, os, random, re, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CRL = os.path.dirname(HERE)
sys.path.insert(0, CRL)
import smol_code as sc  # noqa: E402
from evaluate import HE_INSTR, build_program, extract_code, run_program  # noqa: E402  (path added by smol_code)

PKG = os.path.dirname(CRL)                          # the v14/ folder
SCR = os.environ.get("ENGINE_SCR", os.path.join(PKG, "scr"))          # holds local/msdata/heldout_prompts.jsonl
MODEL = os.environ.get("ENGINE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
DEV = os.environ.get("ENGINE_DEVICE", "cpu")       # "cuda" on Kaggle; everything then runs in float32 on the GPU
if DEV != "cpu":
    torch.set_default_device(DEV)
MBPP_INSTR = sc.MBPP_INSTR
CLOSED = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)
T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} {time.time() - T0:7.0f}s]", *a, flush=True)


def load(path=None, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = path or MODEL
    if DEV != "cpu":
        dtype = torch.float32                      # T4: no fast bf16; the model is small, so float32 throughout
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(DEV).eval()
    return tok, model


def chat_ids(tok, user):
    return sc.chat_ids(tok, user)


def mbpp_prompt(p):
    return MBPP_INSTR.format(text=p["text"], test=p["test_list"][0])


def problems(split):
    return sc.mbpp(split)


# ------------------------------------------------------------------------------------------------ sampling
@torch.inference_mode()
def sample(model, tok, prompts, n, max_new, temperature=1.0, seed=0, top_p=1.0, rows_per_chunk=72, stop_fence=True,
           return_logps=False):
    """prompts: list of token-id lists. Returns out[i] = list of n completions (token-id lists), cut at the stop point
    (the end token is kept when the row ended with one). With return_logps, also lps[i] = per-completion tensors of the
    sampling log-probability of every token (the behaviour policy, for importance weights)."""
    from transformers import DynamicCache
    eos = set(sc.eos_ids(tok))
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    per_chunk = max(1, rows_per_chunk // n)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    out_lp = [None] * len(prompts)
    dev_dtype = next(model.parameters()).dtype
    for s in range(0, len(order), per_chunk):
        grp = order[s:s + per_chunk]
        L = max(len(prompts[i]) for i in grp)
        ids = torch.full((len(grp), L), pad, dtype=torch.long)
        att = torch.zeros((len(grp), L), dtype=torch.long)
        for j, i in enumerate(grp):
            ids[j, L - len(prompts[i]):] = torch.tensor(prompts[i])
            att[j, L - len(prompts[i]):] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        cache = DynamicCache()
        o = model(input_ids=ids, attention_mask=att, position_ids=pos, past_key_values=cache, use_cache=True,
                  logits_to_keep=1)
        cache = o.past_key_values
        rep = torch.arange(len(grp)).repeat_interleave(n)
        cache.batch_repeat_interleave(n)
        logits = o.logits[:, -1].float().repeat_interleave(n, 0)
        att = att.repeat_interleave(n, 0)
        R = len(rep)
        toks = [[] for _ in range(R)]
        lps = [[] for _ in range(R)]
        texts = [""] * R
        live = torch.arange(R)                       # row ids of the rows still in the batch
        for step in range(max_new):
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                pr = torch.softmax(logits / temperature, -1)
                if top_p < 1.0:
                    sp, si = pr.sort(-1, descending=True)
                    cut = sp.cumsum(-1) - sp > top_p
                    sp[cut] = 0
                    pr = torch.zeros_like(pr).scatter_(-1, si, sp)
                nxt = torch.multinomial(pr, 1, generator=gen)[:, 0]
            if return_logps:
                tlp = torch.log_softmax(logits / max(temperature, 1e-6), -1).gather(1, nxt[:, None])[:, 0].tolist()
            keep = []
            for k, r in enumerate(live.tolist()):
                t = int(nxt[k])
                toks[r].append(t)
                if return_logps:
                    lps[r].append(tlp[k])
                done = t in eos or step == max_new - 1
                if not done and stop_fence:
                    piece = tok.decode([t])
                    texts[r] += piece
                    if "`" in piece and CLOSED.search(texts[r]):
                        done = True
                if not done:
                    keep.append(k)
            if not keep:
                break
            if len(keep) < len(live):
                kt = torch.tensor(keep)
                cache.batch_select_indices(kt)
                att = att[kt]
                nxt = nxt[kt]
                live = live[kt]
            att = torch.cat([att, torch.ones((len(live), 1), dtype=att.dtype)], 1)
            pos = (att.sum(-1, keepdim=True) - 1)
            o = model(input_ids=nxt[:, None], attention_mask=att, position_ids=pos, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            logits = o.logits[:, -1].float()
        for j, i in enumerate(grp):
            out[i] = [toks[j * n + q] for q in range(n)]
            out_lp[i] = [torch.tensor(lps[j * n + q]) for q in range(n)]
    return (out, out_lp) if return_logps else out


# ------------------------------------------------------------------------------------------------ grading
def grade(text, prob, timeout=3):
    """1 if the answer's function passes every test of the MBPP problem, else 0."""
    code = extract_code(text)
    if not re.search(r"^\s*def\s+\w+\s*\(", code, re.M):
        return 0
    res = sc.run_tests(code, prob["test_list"], prob["test_setup_code"] or "", timeout=timeout)
    return int(isinstance(res, tuple) and res[0] == res[1])


def grade_many(texts, probs, workers=4):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda tp: grade(*tp), zip(texts, probs)))


# ------------------------------------------------------------------------------------------------ log-probs
def token_logps(model, seqs, plens, pad, micro=8):
    """Per-sequence tensors of the log-probs of the completion tokens (with grad if enabled)."""
    res = []
    for s in range(0, len(seqs), micro):
        mb, mp = seqs[s:s + micro], plens[s:s + micro]
        L = max(len(x) for x in mb)
        ids = torch.full((len(mb), L), pad, dtype=torch.long)
        att = torch.zeros((len(mb), L), dtype=torch.long)
        for j, x in enumerate(mb):
            ids[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
        W = model.get_output_embeddings().weight
        for j, (x, p) in enumerate(zip(mb, mp)):
            hj = h[j, p - 1:len(x) - 1]
            lg = (hj @ W.T).float()
            tgt = torch.tensor(x[p:])
            res.append(-F.cross_entropy(lg, tgt, reduction="none"))
    return res


# ------------------------------------------------------------------------------------------------ checkpoints
def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_ckpt(out, state):
    """Keep two checkpoints: ckpt_a / ckpt_b alternate, latest.json names the newest complete one."""
    lj = os.path.join(out, "latest.json")
    prev = json.load(open(lj))["name"] if os.path.exists(lj) else "ckpt_b.pt"
    name = "ckpt_a.pt" if prev == "ckpt_b.pt" else "ckpt_b.pt"
    atomic_save(state, os.path.join(out, name))
    tmp = lj + ".tmp"
    json.dump(dict(name=name, round=state["round"], compute_s=state["compute_s"]), open(tmp, "w"))
    os.replace(tmp, lj)


def load_ckpt(out):
    lj = os.path.join(out, "latest.json")
    if not os.path.exists(lj):
        return None
    return torch.load(os.path.join(out, json.load(open(lj))["name"]), weights_only=False)


# ------------------------------------------------------------------------------------------------ bench
if __name__ == "__main__":
    tok, model = load()
    probs = problems("train")[:12]
    prompts = [chat_ids(tok, mbpp_prompt(p)) for p in probs]
    for dt in (torch.float32, torch.bfloat16):
        m = model.to(dt)
        t = time.time()
        outs = sample(m, tok, prompts, 6, 160, 1.0, seed=0)
        dt_s = time.time() - t
        flat = [x for o in outs for x in o]
        texts = [tok.decode(x, skip_special_tokens=True) for x in flat]
        g = grade_many(texts, [p for p in probs for _ in range(6)])
        log(f"{dt}: {len(flat)} samples in {dt_s:.1f}s, mean len {sum(map(len, flat)) / len(flat):.0f}, "
            f"pass {sum(g)}/{len(g)}, problems with a pass {sum(any(g[i*6:(i+1)*6]) for i in range(12))}/12")
    print(texts[0][:400])
