# LLM → NCM conversion report: /tmp/SmolLM2-135M-Instruct-textonly

Block = 32 tokens, code levels [2, 8] (codebook 1000), prefix 64, 24000 training windows (50% are the original model's own answers to chat prompts; 100% of the rest distilled from it).
Generator backbone: **llm** (full attention), 5000 steps, batch 8.
Passes per block = 3 code passes (first level written one slot at a time) + R token passes (+1 to cache the finished block); the original model needs 32 passes per block.

## Quality and speed vs the original model

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.546 | 1.0 | 0.688 | 0.076 | 29.8 | 1.016 | 2/20 |
| NCM refine=1 | 4.73 | 0.113 | 0.226 | 0.049 | 106.5 | 0.156 | - |
| NCM refine=2 | 5.008 | 0.107 | 0.312 | 0.076 | 84.8 | 0.188 | - |
| NCM refine=4 | 5.331 | 0.097 | 0.351 | 0.108 | 66.2 | 0.25 | 0/20 |
| real held-out text (reference) | 2.832 | - | - | - | - | - | - |

Judge bits/token: the original model's surprise at each output, lower = closer to what it would write. The original's own greedy output is the floor; real held-out text shows what normal human-written text scores.

## Compressor

Token reconstruction accuracy from codes alone (held-out): 1 level(s): 9.8%, 2 level(s): 20.4%; codebook use per level: [0.485, 0.848].

## Sample outputs (first prompt)

**original (token-by-token)**

```
 get_user_info(user_info):
    user_info_str = user_info.decode('utf-8')
    user_info_str = user_info_str.strip()
    user_info_str = user_info_str.replace(' ', '')
    user_info_
```

**NCM refine=1**

```
 get___():
   
   
   (__
   
   
   
   
   
   _________.___(self,______________________________
```

**NCM refine=2**

```
 get___():
   
   
    return____
   __
   _____________name =
   ______________________________
```

**NCM refine=4**

```
 get__error_
   
   
   
    if___name =
   _____name_
   
    return______name
   ______________________________
```

