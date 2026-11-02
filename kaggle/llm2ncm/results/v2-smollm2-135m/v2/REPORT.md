# LLM → NCM conversion report: /tmp/SmolLM2-135M-Instruct-textonly

Block = 32 tokens, code levels [2, 8] (codebook 1024), prefix 64, 24000 training windows (50% are the original model's own answers to chat prompts; 100% of the rest distilled from it).
Generator backbone: **llm** (full attention), 5000 steps, batch 8.
Passes per block = 2 code levels + R token passes (+1 to cache the finished block); the original model needs 32 passes per block.

## Quality and speed vs the original model

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.546 | 1.0 | 0.688 | 0.076 | 29.2 | 1.016 | 2/20 |
| NCM refine=1 | 4.559 | 0.087 | 0.27 | 0.0 | 125.7 | 0.127 | - |
| NCM refine=2 | 5.086 | 0.098 | 0.334 | 0.059 | 100.0 | 0.158 | - |
| NCM refine=4 | 5.237 | 0.084 | 0.384 | 0.032 | 73.9 | 0.221 | 0/20 |
| real held-out text (reference) | 2.832 | - | - | - | - | - | - |

Judge bits/token: the original model's surprise at each output, lower = closer to what it would write. The original's own greedy output is the floor; real held-out text shows what normal human-written text scores.

## Compressor

Token reconstruction accuracy from codes alone (held-out): 1 level(s): 11.3%, 2 level(s): 22.5%; codebook use per level: [0.334, 0.708].

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
____((,,,,,,,,,,,,,,___,,_,_____,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
```

**NCM refine=2**

```
____((branch, name,,,name, name,,,,,___,,,,_____name,,,,,_,,,,,name,,,,,,_name,,,,,,,,,,,
```

**NCM refine=4**

```
____branch(branch, name,,,name,name, path,,,__):,,__, int___path(name,,,____name,,,name,,,,,name,,,____name,,,
```

