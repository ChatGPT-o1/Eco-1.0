# LLM → NCM conversion report: /tmp/SmolLM2-135M-Instruct-textonly

Block = 32 tokens, code levels [4, 16] (codebook 1000), prefix 64, 24000 training windows (50% are the original model's own answers to chat prompts; 100% of the rest distilled from it).
Generator backbone: **llm** (full attention), 5000 steps, batch 8.
Passes per block = 5 code passes (first level written one slot at a time) + R token passes (+1 to cache the finished block); the original model needs 32 passes per block.

## Quality and speed vs the original model

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.546 | 1.0 | 0.688 | 0.076 | 30.4 | 1.016 | 2/20 |
| NCM refine=1 | 4.436 | 0.101 | 0.283 | 0.022 | 78.5 | 0.219 | - |
| NCM refine=2 | 4.796 | 0.092 | 0.375 | 0.058 | 67.7 | 0.25 | - |
| NCM refine=4 | 4.425 | 0.078 | 0.366 | 0.018 | 55.0 | 0.312 | 0/20 |
| real held-out text (reference) | 2.832 | - | - | - | - | - | - |

Judge bits/token: the original model's surprise at each output, lower = closer to what it would write. The original's own greedy output is the floor; real held-out text shows what normal human-written text scores.

## Compressor

Token reconstruction accuracy from codes alone (held-out): 1 level(s): 12.1%, 2 level(s): 32.9%; codebook use per level: [0.76, 0.964].

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
 get__(____________________________________________________________
```

**NCM refine=2**

```
 is_split(branch__sv_str_________branch_________________branch_branch_branch_branch____________________
```

**NCM refine=4**

```
 is_valid(branch__type_str_____type_type_____branch________branch__________type_type_type_type_type_____type______
```

