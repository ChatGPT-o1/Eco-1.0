# LLM → NCM conversion report: /tmp/SmolLM2-135M-Instruct-textonly

Block = 32 tokens, code levels [2, 8] (codebook 1024), prefix 64, 50000 training windows (50% distilled from the original model).
Generator backbone: **llm** (full attention), 5000 steps, batch 8.
Passes per block = 2 code levels + R token passes (+1 to cache the finished block); the original model needs 32 passes per block.

## Quality and speed vs the original model

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.546 | 1.0 | 0.688 | 0.076 | 31.3 | 1.016 | 2/20 |
| NCM refine=1 | 4.652 | 0.067 | 0.195 | 0.0 | 138.7 | 0.125 | - |
| NCM refine=2 | 4.692 | 0.064 | 0.21 | 0.0 | 113.6 | 0.156 | - |
| NCM refine=4 | 4.768 | 0.06 | 0.23 | 0.072 | 80.7 | 0.219 | 0/20 |
| real held-out text (reference) | 2.832 | - | - | - | - | - | - |

Judge bits/token: the original model's surprise at each output, lower = closer to what it would write. The original's own greedy output is the floor; real held-out text shows what normal human-written text scores.

## Compressor

Token reconstruction accuracy from codes alone (held-out): 1 level(s): 9.6%, 2 level(s): 20.4%; codebook use per level: [0.323, 0.781].

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
 get__((((,((,):):
   
    print ' '\\\\nnn\\\\\\\n\\nnnnnnnnnnnnnnnnnnnnnnnnnnnnn
```

**NCM refine=2**

```
 get__((branch,,,(,):):
   
    print(' '\\\\nn\\\\\\\\n\\\\\\nn\\\\\\\\\\\n\\\\\\n\\\\
```

**NCM refine=4**

```
 get__((branch,,((,):):
   
    print(' '\\\\nn\\\\\\\\n\\\\\\nn\\n\\\n\\\nn\\nn\\n\\\n
```

