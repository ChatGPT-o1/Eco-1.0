#!/usr/bin/env python3
"""big_chat.py - the typed chatbot scaled up: 10 Haiku agents hand-typed the listening weights and replies in blocks/. No data, no training.

    python handmade_chat.py            # chat in the terminal

The editor wrote the knowledge (KNOWLEDGE below: what to listen for and what to say). The script turns it into the
weights of a standard one-layer transformer and then talks with it by ordinary next-token generation:

    tokens -> embedding -> attention -> MLP -> unembedding -> argmax next token

How each part is written:
  embedding    one-hot: every token has its own direction in the residual stream
  attention    one head, looking evenly at the user's words. Its value matrix holds the listening knowledge:
               each word votes for the topics it signals ("france" -> capital of France, "hole" -> black holes ...),
               and the head writes the vote totals into topic slots of the residual stream
  MLP          two kinds of hand-written neurons
                 topic neurons:  topic slot -> "start that topic's answer" (the first word of the reply)
                 phrase neurons: "this word of an answer" -> "the next word of that answer" (with a large weight, so a
                                 reply, once started, is finished in order), and the last word -> <end>
  unembedding  reads the next-token slots
  fallback     an output bias that says "I don't know that one yet" when no word voted for anything

Limits by construction: it only knows what is in KNOWLEDGE, it reads each message on its own (no memory of earlier
turns), and it matches words, not meaning.
"""
import re
import numpy as np

# ----------------------------------------------------------------------------------------- the knowledge (hand-written)
# Each topic: words to listen for (with how strongly each signals the topic) and the reply.
import glob, json, os
KNOWLEDGE = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blocks', '*.json'))):
    for t in json.load(open(f))['topics']:
        KNOWLEDGE.append(({w.lower(): float(v) for w, v in t['listen'].items()}, ' '.join(t['reply'])))
FALLBACK = "I don't know that one yet. My knowledge was written by hand, so it's small."

# ----------------------------------------------------------------------------------------- vocabulary
def words(text): return re.findall(r"[a-z0-9']+", text.lower())

UNK, BOT, END = 0, 1, 2
names = ['<unk>', '<bot>', '<end>']
user_id = {}
for kw, _ in KNOWLEDGE:
    for w in kw:
        if w not in user_id: user_id[w] = len(names); names.append(w)
topics = [(kw, reply) for kw, reply in KNOWLEDGE] + [({}, FALLBACK)]
NI = len(topics); FB = NI - 1
first, nxt, shown = [], {}, {}          # first bot token of each topic; next-token map; display text of bot tokens
for ti, (_, reply) in enumerate(topics):
    ids = []
    for piece in reply.split(' '):
        tid = len(names); names.append(f'<{ti}:{len(ids)}>'); shown[tid] = piece; ids.append(tid)
    first.append(ids[0])
    for a, b in zip(ids, ids[1:]): nxt[a] = b
    nxt[ids[-1]] = END
NV = len(names)

# ----------------------------------------------------------------------------------------- weights (hand-written)
TOK, TOP, OUT = 0, NV, NV + NI                 # residual layout: token one-hot | topic votes | next-token scores
D = NV + NI + NV
SCALE, BIG, FALLBACK_BIAS = 100.0, 1000.0, 0.5

W_E = np.zeros((NV, D)); W_E[np.arange(NV), TOK + np.arange(NV)] = 1                 # embedding
W_Q = np.zeros((D, 1)); W_K = np.zeros((D, 1))                                        # all scores 0: even attention
W_V = np.zeros((D, NI)); W_O = np.zeros((NI, D))
for ti, (kw, _) in enumerate(topics):
    for w, s in kw.items(): W_V[TOK + user_id[w], ti] = s                            # word -> topic votes
    W_O[ti, TOP + ti] = SCALE
H = NI + NV                                                                           # MLP neurons: topic neurons, phrase neurons
W_1 = np.zeros((D, H)); b_1 = np.zeros(H); W_2 = np.zeros((H, D))
for ti in range(NI):
    W_1[TOP + ti, ti] = 1; W_2[ti, OUT + first[ti]] = 1                              # topic -> its first reply word
for a, b in nxt.items():
    W_1[TOK + a, NI + a] = 1; W_2[NI + a, OUT + b] = BIG                              # reply word -> next reply word
W_U = np.zeros((D, NV)); W_U[OUT + np.arange(NV), np.arange(NV)] = 1                # unembedding
b_U = np.zeros(NV); b_U[first[FB]] = FALLBACK_BIAS
b_U[:] += np.where([n in shown or n in (END,) for n in range(NV)], 0, -1e9)          # only reply words or <end> can be said

# ----------------------------------------------------------------------------------------- the transformer
def forward(ids):
    x = W_E[ids]                                                                      # (T, D)
    q, k, v = x @ W_Q, x @ W_K, x @ W_V
    s = q @ k.T; s = s + np.triu(np.full(s.shape, -1e9), 1)                           # causal
    a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
    x = x + (a @ v) @ W_O
    x = x + np.maximum(0, x @ W_1 + b_1) @ W_2
    return x @ W_U + b_U                                                              # logits (T, NV)

def reply(message, max_len=60):
    ids = [user_id.get(w, UNK) for w in words(message)] + [BOT]
    out = []
    for _ in range(max_len):
        t = int(forward(np.array(ids))[-1].argmax())
        if t == END: break
        ids.append(t); out.append(shown[t])
    return ' '.join(out)

if __name__ == '__main__':
    n_params = sum(m.size for m in (W_E, W_Q, W_K, W_V, W_O, W_1, b_1, W_2, W_U, b_U))
    nonzero = sum(int(np.count_nonzero(np.isfinite(m) & (m != 0))) for m in (W_E, W_V, W_O, W_1, W_2, W_U))
    print(f'hand-written transformer: {n_params:,} parameters ({nonzero:,} non-zero), {NV} tokens, {NI} topics. Type "quit" to stop.')
    while True:
        try: msg = input('you: ')
        except EOFError: break
        if msg.strip().lower() in ('quit', 'exit'): break
        print('bot:', reply(msg))
