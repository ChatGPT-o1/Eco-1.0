"""judge.py - the scoring code. The agent may NOT edit this file or the data it reads.

Data (all wikitext-103, disjoint by source file and by article):
    search      validation-split articles, even-numbered   (agent's reward)
    validation  validation-split articles, odd-numbered    (gate before a new base is accepted)
    test        test split                                 (final report only)
    healing     train split, first shard                   (distillation text)
Speed: greedy generation, batch 1, KV cache, fp32, fixed thread count; median of many timed runs.
"""
import os, re, time, statistics, functools
import numpy as np, torch, pandas as pd

ROOT = os.environ.get('PRUNE_ROOT', '/tmp/claude-0/-home-user-Eco-1-0/44fe88f8-9d28-5d7b-9454-d407264e70f1/scratchpad/prune')
MODEL_DIR = f'{ROOT}/model'
DATA = f'{ROOT}/data/wikitext-103-raw-v1'
SEQ = 256
SIZES = {'search': 32, 'validation': 64, 'test': 128}
DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
THREADS = 3                     # 4 threads on this VM are noisy (one straggler vCPU stalls every matmul)
torch.set_num_threads(THREADS)

def _articles(path):
    arts, cur = [], []
    for row in pd.read_parquet(path)['text'].tolist():
        if row.startswith(' = ') and not row.startswith(' = = ') and cur:
            arts.append(''.join(cur)); cur = []
        cur.append(row)
    if cur: arts.append(''.join(cur))
    return [a for a in arts if len(a) > 2000]

@functools.lru_cache(None)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)

def _chunks(articles, n):
    tok = tokenizer(); out = []
    for a in articles:
        ids = tok(a)['input_ids']
        for i in range(0, len(ids) - SEQ, SEQ):
            out.append(ids[i:i + SEQ])
            if len(out) == n: return torch.tensor(out)
    raise RuntimeError('not enough text')

@functools.lru_cache(None)
def split(name):
    if name in ('search', 'validation'):
        arts = _articles(f'{DATA}/validation-00000-of-00001.parquet')
        arts = arts[0::2] if name == 'search' else arts[1::2]
        rng = np.random.default_rng(1); rng.shuffle(arts)
        return _chunks(arts, SIZES[name])
    if name == 'test':
        arts = _articles(f'{DATA}/test-00000-of-00001.parquet')
        rng = np.random.default_rng(2); rng.shuffle(arts)
        return _chunks(arts, SIZES['test'])
    if name == 'healing':
        arts = _articles(f'{DATA}/train-00000-of-00002.parquet')[:4000]
        return _chunks(arts, 4000)
    raise KeyError(name)

@torch.no_grad()
def perplexity(model, name, batch=16):
    x = split(name); nll, n = 0.0, 0
    model.eval()
    for i in range(0, len(x), batch):
        b = x[i:i + batch].to(DEVICE)
        logits = model(b).logits[:, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1), reduction='sum')
        nll += loss.item(); n += b[:, 1:].numel()
    return float(np.exp(nll / n))

PROMPTS = ['The history of the city began in', 'In 1998, the band released', 'The species is found in', 'During the war, the army']

@torch.no_grad()
def speed(model, runs=20, new_tokens=64, warmup=2):
    """Median tokens/second for greedy generation (batch 1, KV cache)."""
    tok = tokenizer(); model.eval(); rates = []
    for r in range(warmup + runs):
        ids = tok(PROMPTS[r % len(PROMPTS)], return_tensors='pt')['input_ids'].to(DEVICE); _sync()
        t0 = time.perf_counter()
        model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=0)
        _sync(); dt = time.perf_counter() - t0
        if r >= warmup: rates.append(new_tokens / dt)
    return statistics.median(rates), statistics.stdev(rates) / len(rates) ** 0.5

def _sync():
    if DEVICE == 'cuda': torch.cuda.synchronize()

def load_original():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to(DEVICE)

@torch.no_grad()
def sample(model, prompt, n=40):
    tok = tokenizer(); ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=0)
    return tok.decode(out[0][ids.shape[1]:])

@torch.no_grad()
def quick_speed(model, runs=6, new_tokens=48):
    """Fast, low-noise speed reading for per-step rewards: same prompt every run, best (minimum) time of N."""
    tok = tokenizer(); model.eval(); ids = tok(PROMPTS[0], return_tensors='pt')['input_ids'].to(DEVICE)
    model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=0)            # warm-up
    best = 1e9
    for _ in range(runs):
        _sync(); t0 = time.perf_counter()
        model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=0)
        _sync(); best = min(best, time.perf_counter() - t0)
    return new_tokens / best
