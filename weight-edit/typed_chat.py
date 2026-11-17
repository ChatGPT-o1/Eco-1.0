#!/usr/bin/env python3
"""typed_chat.py - a tiny transformer chatbot. Every weight below was typed by hand, one number at a time.
No tables, no loops that make weights, no data, no training. The only code is the standard forward pass.

    python typed_chat.py

It knows three things: greetings, the capital of France, and that you can't fall into... become a black hole.
Anything else gets "Sorry?".

Residual stream (7 numbers per token):
    r0 greeting vote | r1 France vote | r2 black-hole vote | r3 reply starts here | r4 word is "No," | r5 word is "you" | r6 word ends a reply
"""
import re
import numpy as np

TOKENS = ["<unk>", "hi", "capital", "france", "hole", "<bot>", "<end>", "Hello!", "Paris.", "No,", "you", "can't.", "Sorry?"]

# embedding: one row per token, 7 columns (r0..r6)
W_E = np.array([
    #  r0   r1   r2   r3   r4   r5   r6
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # <unk>
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # hi
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],   # capital
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # france
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],   # hole
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # <bot>
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # <end>
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # Hello!
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # Paris.
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   # No,
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # you
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # can't.
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # Sorry?
])

# attention: query and key are all zeros, so the head looks at every earlier word equally
W_Q = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
W_K = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
# value: copy the three votes
W_V = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])
# output: write the averaged votes back, ten times stronger
W_O = np.array([
    [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
])

# unembedding: 7 rows (r0..r6), one column per token
W_U = np.array([
    # <unk> hi  capital france hole <bot> <end> Hello! Paris. No,  you  can't. Sorry?
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0,  0.0, 0.0],   # r0 greeting  -> "Hello!"
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0,  0.0, 0.0],   # r1 France    -> "Paris."
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0,  0.0,  0.0, 0.0],   # r2 black hole -> "No,"
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0],   # r3
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 20.0,  0.0, 0.0],   # r4 after "No,"  -> "you"
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 20.0, 0.0],   # r5 after "you"  -> "can't."
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0],   # r6 reply ended  -> <end>
])
# output bias: never say a listening word; say "Sorry?" if nothing voted
b_U = np.array([-100.0, -100.0, -100.0, -100.0, -100.0, -100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05])

# ------------------------------------------------------------------ the model (standard forward pass, no weights made here)
def forward(ids):
    x = W_E[ids]
    s = (x @ W_Q) @ (x @ W_K).T + np.triu(np.full((len(ids), len(ids)), -1e9), 1)
    a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
    x = x + (a @ (x @ W_V)) @ W_O
    return x @ W_U + b_U

def reply(message):
    ids = [TOKENS.index(w) if w in TOKENS[1:5] else 0 for w in re.findall(r"[a-z]+", message.lower())] + [5]
    out = []
    for _ in range(10):
        t = int(forward(np.array(ids))[-1].argmax())
        if t == 6: break
        ids.append(t); out.append(TOKENS[t])
    return ' '.join(out)

if __name__ == '__main__':
    print('typed-by-hand transformer: 202 numbers. Type "quit" to stop.')
    while True:
        try: msg = input('you: ')
        except EOFError: break
        if msg.strip().lower() == 'quit': break
        print('bot:', reply(msg))
