# LLM → NCM conversion report: /tmp/SmolLM2-135M-Instruct-textonly

Block = 32 tokens, code levels [4, 16] (codebook 1024), prefix 64, 24000 training windows (50% are the original model's own answers to chat prompts; 100% of the rest distilled from it).
Generator backbone: **llm** (full attention), 5000 steps, batch 8.
Passes per block = 2 code levels + R token passes (+1 to cache the finished block); the original model needs 32 passes per block.

## Quality and speed vs the original model

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.546 | 1.0 | 0.688 | 0.076 | 29.8 | 1.016 | 2/20 |
| NCM refine=1 | 5.065 | 0.07 | 0.367 | 0.075 | 131.1 | 0.125 | - |
| NCM refine=2 | 4.832 | 0.074 | 0.408 | 0.176 | 107.7 | 0.156 | - |
| NCM refine=4 | 4.92 | 0.083 | 0.439 | 0.03 | 75.3 | 0.219 | 0/20 |
| real held-out text (reference) | 2.832 | - | - | - | - | - | - |

Judge bits/token: the original model's surprise at each output, lower = closer to what it would write. The original's own greedy output is the floor; real held-out text shows what normal human-written text scores.

## Compressor

Token reconstruction accuracy from codes alone (held-out): 1 level(s): 13.5%, 2 level(s): 43.1%; codebook use per level: [0.515, 0.836].

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
 get_branch_string((, text):
   
   ::
        else: else: else: else: return return '\ '\ 'ERROR_:[]]
           \\\\\m'
        return '\ '\ '\ 'EN__ERROR: '\\ ' '\\\::[
```

**NCM refine=2**

```
 get_branch_string(string, text):
   
    return:
        else: else: else: else: return return '\ '\ 'ERROR_:[]]
           \\\\\\\\\\\\\\\\\\\\\\\\\\\\\
```

**NCM refine=4**

```
 get_branch_branch(string, text):
   
    return:
        else: else
        else
        else: return return '\ '\ 'ERROR_:[]]
       text[text[text[text]\\\n\[[[\\\\\\\\\\\\\
```

