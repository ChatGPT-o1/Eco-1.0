"""Vocabulary surgery for byte-level BPE tokenizers (tokenizer.json): keep a subset of tokens
such that any text whose original tokenization used only kept tokens tokenizes identically.

Key idea: a BPE token can only be produced if the two parents of the merge that creates it can
be produced, so the kept set is closed under "merge parents" (plus all 256 byte tokens and all
added/special tokens). Text that needed a dropped token still encodes, just as a few more pieces.
"""
import json
import os

import numpy as np


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def load_tokenizer_json(model_dir):
    with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as f:
        return json.load(f)


def _has_bytelevel(node):
    if not node:
        return False
    if node.get("type") == "ByteLevel":
        return True
    return any(_has_bytelevel(x) for x in node.get("pretokenizers", []) or [])


def check_supported(tj):
    """Raise SystemExit with a reason if this tokenizer can't be pruned safely."""
    m = tj.get("model", {})
    if m.get("type") != "BPE":
        raise SystemExit(f"vocab pruning needs a BPE tokenizer.json (got model.type={m.get('type')!r})")
    if m.get("byte_fallback"):
        raise SystemExit("SentencePiece-style BPE with byte_fallback is not supported (Llama-2/Mistral-v1/Gemma style)")
    if not _has_bytelevel(tj.get("pre_tokenizer")):
        raise SystemExit("needs a ByteLevel pre-tokenizer (GPT-2 / Llama-3 / Qwen style)")


def _merge_pairs(merges):
    out = []
    for m in merges:
        if isinstance(m, str):
            a, b = m.split(" ", 1)
        else:
            a, b = m
        out.append((a, b))
    return out


class Vocab:
    """Parsed view of a tokenizer.json for pruning."""

    def __init__(self, tj):
        check_supported(tj)
        self.tj = tj
        self.vocab = tj["model"]["vocab"]                      # token -> id
        self.merges = _merge_pairs(tj["model"]["merges"])
        self.added = sorted(tj.get("added_tokens", []), key=lambda a: a["id"])
        ids = list(self.vocab.values()) + [a["id"] for a in self.added]
        self.n_ids = max(ids) + 1                              # id space covered by the tokenizer
        self.n_model = max(self.vocab.values()) + 1
        self.id2tok = {i: t for t, i in self.vocab.items()}
        base = list(bytes_to_unicode().values())
        present = [c for c in base if c in self.vocab]
        if len(present) < 200:
            raise SystemExit(f"only {len(present)}/256 byte-level base tokens in vocab; not a byte-level BPE")
        if len(present) < 256:  # e.g. SmolLM2: unused control bytes were never in the vocab, so nothing changes
            import sys
            print(f"note: {256 - len(present)} byte tokens absent from the original vocab (unencodable before and after)", file=sys.stderr)
        self.base_ids = np.array([self.vocab[c] for c in present])
        self.added_ids = np.array([a["id"] for a in self.added], dtype=np.int64)
        # (parent_a, parent_b, result) ids; merges whose pieces aren't all in vocab can never fire
        trip = []
        for a, b in self.merges:
            x, y, r = self.vocab.get(a, -1), self.vocab.get(b, -1), self.vocab.get(a + b, -1)
            trip.append((x, y, r))
        self.trip = np.array(trip, dtype=np.int64)

    def keep_mask(self, counts, min_count, extra_ids=()):
        """Seed = tokens with count >= min_count, base bytes, added tokens, extra ids; then close under merges."""
        keep = np.zeros(self.n_ids, dtype=bool)
        c = np.zeros(self.n_ids, dtype=np.int64)
        c[: min(len(counts), self.n_ids)] = counts[: self.n_ids]
        keep |= c >= min_count
        keep[self.base_ids] = True
        if len(self.added_ids):
            keep[self.added_ids] = True
        for i in extra_ids:
            keep[i] = True
        # ids not used by the tokenizer at all (embedding padding rows) are never kept
        valid = np.zeros(self.n_ids, dtype=bool)
        valid[list(self.vocab.values())] = True
        if len(self.added_ids):
            valid[self.added_ids] = True
        keep &= valid
        t = self.trip[(self.trip >= 0).all(axis=1)]
        while True:  # fixed point: kept result => both parents kept
            need = keep[t[:, 2]] & ~(keep[t[:, 0]] & keep[t[:, 1]])
            if not need.any():
                break
            keep[t[need, 0]] = True
            keep[t[need, 1]] = True
        return keep

    def build(self, keep):
        """Return (new tokenizer.json dict, old2new int array, kept old ids in new order)."""
        model_ids = sorted(i for i in self.vocab.values() if keep[i])
        added_only = [a["id"] for a in self.added if a["id"] not in self.id2tok]
        order = model_ids + added_only
        old2new = -np.ones(self.n_ids, dtype=np.int64)
        old2new[np.array(order, dtype=np.int64)] = np.arange(len(order))
        tj = json.loads(json.dumps(self.tj))
        new_vocab = {self.id2tok[i]: int(old2new[i]) for i in model_ids}
        raw = self.tj["model"]["merges"]
        new_merges = [m for m, (x, y, r) in zip(raw, self.trip) if r >= 0 and keep[r] and x >= 0 and y >= 0]
        tj["model"]["vocab"] = new_vocab
        tj["model"]["merges"] = new_merges
        tj["added_tokens"] = [dict(a, id=int(old2new[a["id"]])) for a in self.added]
        return tj, old2new, np.array(order, dtype=np.int64)


def tokenizer_from_json(tj):
    from tokenizers import Tokenizer
    return Tokenizer.from_str(json.dumps(tj))


def inflation(old_tok, new_tok, texts):
    """Total-token inflation and fraction of texts tokenized to the same number of tokens."""
    if not texts:
        return 0.0, 1.0
    o = old_tok.encode_batch(texts, add_special_tokens=False)
    n = new_tok.encode_batch(texts, add_special_tokens=False)
    so = sum(len(e.ids) for e in o)
    sn = sum(len(e.ids) for e in n)
    same = sum(len(a.ids) == len(b.ids) for a, b in zip(o, n))
    return sn / max(so, 1) - 1.0, same / len(texts)


def remap_ids(obj, old2new, old_vocab_size, new_vocab_size, warn):
    """Recursively update *_token_id(s) and vocab_size fields in a config dict."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "vocab_size" and v == old_vocab_size:
                out[k] = new_vocab_size
            elif (k.endswith("_token_id") or k.endswith("_token_ids")) and v is not None:
                out[k] = _map_id(v, old2new, k, warn)
            else:
                out[k] = remap_ids(v, old2new, old_vocab_size, new_vocab_size, warn)
        return out
    if isinstance(obj, list):
        return [remap_ids(v, old2new, old_vocab_size, new_vocab_size, warn) for v in obj]
    return obj


def _map_id(v, old2new, key, warn):
    if isinstance(v, list):
        out = [_map_id(x, old2new, key, warn) for x in v]
        return [x for x in out if x is not None]
    if isinstance(v, int) and not isinstance(v, bool):
        if 0 <= v < len(old2new) and old2new[v] >= 0:
            return int(old2new[v])
        warn(f"{key}={v} refers to a token that was removed or is outside the tokenizer; set to null")
        return None
    return v
