#!/usr/bin/env python3
r"""typed_big_chat.py - a two-layer transformer chatbot. Every weight is a number typed by hand in typed_blocks/*.txt.
This file only places each typed number into its matrix by name and runs the standard forward pass.
It creates no weights: anything not typed is 0.

Line format (an optional leading "L2" puts the weight in layer 2; default layer 1):
    E <token> <slot> <v>      embedding                 W_E[token, slot] = v
    P <pos> <slot> <v>        position embedding        W_P[pos, slot] = v
    V <slot> <topic> <v>      head A value (uniform attention: Q=K=0)
    O <topic> <slot> <v>      head A output
    QB <slot> <dim> <v>       head B query   \  a second attention head whose scores
    KB <slot> <dim> <v>       head B key      } come from typed position numbers
    VB <slot> <dim> <v>       head B value   /  (e.g. "look at the previous word")
    OB <dim> <slot> <v>       head B output
    W1 <slot> <neuron> <v>    MLP in
    G  <neuron> <v>           MLP neuron bias (AND-gates)
    W2 <neuron> <slot> <v>    MLP out
    U <slot> <token> <v>      unembedding
    B <token> <v>             output bias
    S <token> <text>          display text of a reply token (not a weight)
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
        else:
            L = 1
            if p[0] == 'L2': L = 2; p = p[1:]
            entries.append((L, p))

def index(): d = {}; return d, (lambda n: d.setdefault(n, len(d)))
tok, T = index(); slot, R = index(); top, P_ = index(); neu, N = index(); dim, D = index()
MAXPOS = 0
for L, p in entries:
    k = p[0]
    if k == 'E': T(p[1]); R(p[2])
    elif k == 'P': MAXPOS = max(MAXPOS, int(p[1]) + 1); R(p[2])
    elif k == 'V': R(p[1]); P_(p[2])
    elif k == 'O': P_(p[1]); R(p[2])
    elif k == 'QB': R(p[1]); D(p[2])
    elif k == 'KB': R(p[1]); D(p[2])
    elif k == 'VB': R(p[1]); D(p[2])
    elif k == 'OB': D(p[1]); R(p[2])
    elif k == 'W1': R(p[1]); N(p[2])
    elif k == 'G': N(p[1])
    elif k == 'W2': N(p[1]); R(p[2])
    elif k == 'U': R(p[1]); T(p[2])
    elif k == 'B': T(p[1])
NT, NR, NP, NN, ND = len(tok), len(slot), len(top), len(neu), max(len(dim), 1)
W_E = np.zeros((NT, NR)); W_P = np.zeros((MAXPOS, NR)) if MAXPOS else None
W_U = np.zeros((NR, NT)); b_U = np.zeros(NT)
Ly = {L: dict(V=np.zeros((NR, NP)), O=np.zeros((NP, NR)), QB=np.zeros((NR, ND)), KB=np.zeros((NR, ND)),
              VB=np.zeros((NR, ND)), OB=np.zeros((ND, NR)), W1=np.zeros((NR, NN)), G=np.zeros(NN),
              W2=np.zeros((NN, NR))) for L in (1, 2)}
for L, p in entries:
    k, v = p[0], float(p[-1]); M = Ly[L]
    if k == 'E': W_E[tok[p[1]], slot[p[2]]] = v
    elif k == 'P': W_P[int(p[1]), slot[p[2]]] = v
    elif k == 'V': M['V'][slot[p[1]], top[p[2]]] = v
    elif k == 'O': M['O'][top[p[1]], slot[p[2]]] = v
    elif k == 'QB': M['QB'][slot[p[1]], dim[p[2]]] = v
    elif k == 'KB': M['KB'][slot[p[1]], dim[p[2]]] = v
    elif k == 'VB': M['VB'][slot[p[1]], dim[p[2]]] = v
    elif k == 'OB': M['OB'][dim[p[1]], slot[p[2]]] = v
    elif k == 'W1': M['W1'][slot[p[1]], neu[p[2]]] = v
    elif k == 'G': M['G'][neu[p[1]]] = v
    elif k == 'W2': M['W2'][neu[p[1]], slot[p[2]]] = v
    elif k == 'U': W_U[slot[p[1]], tok[p[2]]] = v
    elif k == 'B': b_U[tok[p[1]]] = v
names = list(tok)

def _attend(s):
    s = s + np.triu(np.full(s.shape, -1e9), 1)
    a = np.exp(s - s.max(1, keepdims=True)); return a / a.sum(1, keepdims=True)

def forward(ids):
    n = len(ids)
    x = W_E[ids].copy()
    if W_P is not None: x[:min(n, MAXPOS)] += W_P[:min(n, MAXPOS)]
    for L in (1, 2):
        M = Ly[L]
        aA = _attend(np.zeros((n, n)))                                   # head A: uniform
        hA = (aA @ (x @ M['V'])) @ M['O']
        aB = _attend((x @ M['QB']) @ (x @ M['KB']).T)                    # head B: typed position scores
        hB = (aB @ (x @ M['VB'])) @ M['OB']
        x = x + hA + hB
        x = x + np.maximum(0, x @ M['W1'] + M['G']) @ M['W2']
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
    print(f'{len(entries):,} hand-typed weights, {NP} topics, 2 layers. Type "quit" to stop.')
    while True:
        try: msg = input('you: ')
        except EOFError: break
        if msg.strip().lower() in ('quit', 'exit'): break
        print('bot:', reply(msg))
