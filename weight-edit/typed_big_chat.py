#!/usr/bin/env python3
"""typed_big_chat.py - a one-layer transformer chatbot. Every weight is a number typed by hand in typed_blocks/*.txt
(core.txt by Claude, one file per subject by 10 Haiku agents). This file only places each typed number in its matrix
by name and runs the standard forward pass. It creates no weights: anything not typed is 0.

Line format in typed_blocks/*.txt:
    E <token> <slot> <v>        embedding        W_E[token, slot] = v
    V <slot> <topic> <v>        attention value  W_V[slot, topic] = v      (word -> topic vote)
    O <topic> <slot> <v>        attention output W_O[topic, slot] = v
    W1 <slot> <neuron> <v>      MLP in           W_1[slot, neuron] = v
    W2 <neuron> <slot> <v>      MLP out          W_2[neuron, slot] = v
    U <slot> <token> <v>        unembedding      W_U[slot, token] = v
    B <token> <v>               output bias      b_U[token] = v
    S <token> <text>            how a reply token is printed (not a weight)
Attention Q and K are all zeros (even attention over the message).
"""
import glob, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
entries, shown = [], {}
for f in sorted(glob.glob(os.path.join(HERE, 'typed_blocks', '*.txt'))):
    for line in open(f):
        p = line.split()
        if not p or p[0].startswith('#'): continue
        if p[0] == 'S': shown[p[1]] = ' '.join(p[2:])
        else: entries.append(p)

def index(): d = {}; return d, (lambda n: d.setdefault(n, len(d)))
tok, T = index(); slot, R = index(); top, P = index(); neu, N = index()
for p in entries:
    k = p[0]
    if k == 'E': T(p[1]); R(p[2])
    elif k == 'V': R(p[1]); P(p[2])
    elif k == 'O': P(p[1]); R(p[2])
    elif k == 'W1': R(p[1]); N(p[2])
    elif k == 'W2': N(p[1]); R(p[2])
    elif k == 'U': R(p[1]); T(p[2])
    elif k == 'B': T(p[1])
W_E = np.zeros((len(tok), len(slot))); W_V = np.zeros((len(slot), len(top))); W_O = np.zeros((len(top), len(slot)))
W_1 = np.zeros((len(slot), len(neu))); W_2 = np.zeros((len(neu), len(slot))); W_U = np.zeros((len(slot), len(tok)))
b_U = np.zeros(len(tok)); W_Q = np.zeros((len(slot), 1)); W_K = np.zeros((len(slot), 1))
for p in entries:
    k, v = p[0], float(p[-1])
    if k == 'E': W_E[tok[p[1]], slot[p[2]]] = v
    elif k == 'V': W_V[slot[p[1]], top[p[2]]] = v
    elif k == 'O': W_O[top[p[1]], slot[p[2]]] = v
    elif k == 'W1': W_1[slot[p[1]], neu[p[2]]] = v
    elif k == 'W2': W_2[neu[p[1]], slot[p[2]]] = v
    elif k == 'U': W_U[slot[p[1]], tok[p[2]]] = v
    elif k == 'B': b_U[tok[p[1]]] = v
names = list(tok)

def forward(ids):
    x = W_E[ids]
    s = (x @ W_Q) @ (x @ W_K).T + np.triu(np.full((len(ids), len(ids)), -1e9), 1)
    a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
    x = x + (a @ (x @ W_V)) @ W_O
    x = x + np.maximum(0, x @ W_1) @ W_2
    return x @ W_U + b_U

def reply(message, max_len=40):
    ids = [tok.get(w, tok['<unk>']) for w in re.findall(r"[a-z0-9']+", message.lower())] + [tok['<bot>']]
    out = []
    for _ in range(max_len):
        t = int(forward(np.array(ids))[-1].argmax())
        if names[t] == '<end>': break
        ids.append(t); out.append(shown.get(names[t], names[t]))
    return ' '.join(out)

if __name__ == '__main__':
    print(f'{len(entries):,} hand-typed weights, {len(top)} topics. Type "quit" to stop.')
    while True:
        try: msg = input('you: ')
        except EOFError: break
        if msg.strip().lower() in ('quit', 'exit'): break
        print('bot:', reply(msg))
