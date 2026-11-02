# LLM → NCM conversion report: /tmp/SmolLM2-135M-Instruct-textonly

Block = 32 tokens, code levels [4, 16] (codebook 1024), prefix 64, 100000 training windows (50% are the original model's own answers to chat prompts; 100% of the rest distilled from it).
Generator backbone: **llm** (full attention), 41163 steps, batch 32.
Passes per block = 2 code passes + R token passes (+1 to cache the finished block); the original model needs 32 passes per block.

## Quality and speed vs the original model

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.518 | 1.0 | 0.568 | 0.078 | 29.4 | 1.016 | 11/40 |
| NCM refine=1 | 5.89 | 0.118 | 0.502 | 0.024 | 129.2 | 0.125 | - |
| NCM refine=2 | 5.695 | 0.12 | 0.525 | 0.03 | 103.6 | 0.156 | - |
| NCM refine=4 | 5.462 | 0.114 | 0.533 | 0.047 | 73.6 | 0.219 | 0/40 |
| real held-out text (reference) | 2.33 | - | - | - | - | - | - |

Judge bits/token: the original model's surprise at each output, lower = closer to what it would write. The original's own greedy output is the floor; real held-out text shows what normal human-written text scores.

## Compressor

Token reconstruction accuracy from codes alone (held-out): 1 level(s): 14.4%, 2 level(s): 51.7%; codebook use per level: [0.788, 0.978].

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
 get_branch(branch__port,,_port_port_port):
    if_ = parser(branch,branch,branch,branch_port_port, ok_port_port_port_port_port_port):
        if else
        if_branch_port_port_port_port
```

**NCM refine=2**

```
 get_branch(branch_port_, branch_port_port_port):
    if_ = parser(branch,port,port,branch_port_port, branch_port_port_port,branch, branch_port):
        if::
        return
        else else
            else
            else
            else
```

**NCM refine=4**

```
 get_branch(branch_portport, port_port_port_port):
    if_ = parser(branch,port,port,port_port_port):
        return not_branch_port,port, port_port):
        else:
       
           
            get_port_portport port port,
```

