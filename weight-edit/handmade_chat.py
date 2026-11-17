#!/usr/bin/env python3
"""handmade_chat.py - a small transformer chatbot whose every weight is written by hand. No data, no training.

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
STATUS: written, never run.
"""
import re
import numpy as np

# ----------------------------------------------------------------------------------------- the knowledge (hand-written)
# Each topic: words to listen for (with how strongly each signals the topic) and the reply.
KNOWLEDGE = [
 ({"hi": 3, "hello": 3, "hey": 3, "greetings": 3}, "Hello! What would you like to talk about?"),
 ({"how": 1, "are": 1, "you": 1, "doing": 2}, "I'm doing well, thanks for asking. How about you?"),
 ({"name": 3, "who": 1, "you": 1, "called": 2}, "I'm a tiny chatbot. Every one of my weights was written by hand."),
 ({"made": 2, "built": 2, "created": 2, "wrote": 2, "who": 1}, "Claude wrote my weights by hand, number by number, with no training."),
 ({"what": 1, "can": 1, "do": 1, "help": 2, "able": 2}, "I can answer simple questions about science, geography, Python and myself."),
 ({"thanks": 3, "thank": 3, "thx": 3}, "You're welcome!"),
 ({"bye": 3, "goodbye": 3, "later": 2, "night": 1}, "Goodbye! Come back any time."),
 ({"joke": 3, "funny": 2, "laugh": 2}, "Why did the programmer quit? Because they didn't get arrays."),
 ({"weather": 3, "rain": 2, "sunny": 2, "temperature": 2}, "I can't check the weather. I don't have internet access."),
 ({"time": 2, "date": 2, "today": 1}, "I don't have a clock, so I can't tell the time."),
 ({"good": 1, "fine": 2, "great": 2, "well": 1}, "Glad to hear it!"),
 ({"sad": 3, "bad": 2, "tired": 2, "upset": 3}, "I'm sorry you feel that way. Taking a short break or talking to a friend can help."),
 # science
 ({"sky": 3, "blue": 2}, "The sky is blue because air scatters blue sunlight more than red sunlight."),
 ({"black": 1, "hole": 2, "holes": 2, "cause": 2, "causes": 2, "form": 2, "made": 1}, "A black hole forms when the core of a very massive star collapses under its own gravity."),
 ({"black": 1, "hole": 2, "collapse": 3, "become": 2, "turn": 2, "i": 1}, "No. A person has nowhere near enough mass to become a black hole."),
 ({"supernova": 3, "supernovae": 3}, "A supernova is the huge explosion of a massive star at the end of its life."),
 ({"sun": 3, "star": 1}, "The Sun is a star about 150 million kilometers from Earth."),
 ({"moon": 3}, "The Moon is about 384,000 kilometers away and takes about 27 days to orbit Earth."),
 ({"gravity": 3}, "Gravity is the pull between masses. It keeps us on the ground and the planets in orbit."),
 ({"water": 2, "boil": 3, "boils": 3}, "At sea level, water boils at 100 degrees Celsius."),
 ({"photosynthesis": 3, "plants": 2, "plant": 2}, "Plants use sunlight to turn water and carbon dioxide into sugar and oxygen."),
 ({"planets": 3, "planet": 2, "solar": 2}, "There are eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune."),
 # geography
 ({"capital": 1, "france": 3}, "The capital of France is Paris."),
 ({"capital": 1, "japan": 3}, "The capital of Japan is Tokyo."),
 ({"capital": 1, "australia": 3}, "The capital of Australia is Canberra."),
 ({"capital": 1, "canada": 3}, "The capital of Canada is Ottawa."),
 ({"capital": 1, "germany": 3}, "The capital of Germany is Berlin."),
 ({"capital": 1, "italy": 3}, "The capital of Italy is Rome."),
 ({"capital": 1, "spain": 3}, "The capital of Spain is Madrid."),
 ({"capital": 1, "brazil": 3}, "The capital of Brazil is Brasilia."),
 ({"capital": 1, "india": 3}, "The capital of India is New Delhi."),
 ({"capital": 1, "china": 3}, "The capital of China is Beijing."),
 ({"largest": 2, "biggest": 2, "ocean": 3}, "The Pacific is the largest ocean."),
 ({"tallest": 2, "highest": 2, "mountain": 3}, "Mount Everest is the tallest mountain, about 8,849 meters high."),
 # Python
 ({"reverse": 3, "string": 2, "python": 1}, "In Python, s[::-1] reverses a string."),
 ({"sort": 3, "list": 1, "sorted": 3, "python": 1}, "Use sorted(items) for a new sorted list, or items.sort() to sort in place."),
 ({"length": 3, "len": 3, "size": 2, "list": 1}, "len(x) gives the length of a list, string or dictionary."),
 ({"loop": 3, "for": 1, "iterate": 3}, "Loop with: for item in items: then indent the body by four spaces."),
 ({"function": 3, "def": 3, "define": 2}, "Define a function with def name(args): and return a value with return."),
 ({"python": 2, "learn": 2, "start": 1}, "Start with variables, lists, loops and functions, and write a small program every day."),
 ({"dictionary": 3, "dict": 3}, "A dictionary maps keys to values, like ages = {'ana': 30}."),
]
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
