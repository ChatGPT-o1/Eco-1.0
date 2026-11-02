# LLM → NCM conversion report: /tmp/SmolLM2-135M-Instruct-textonly

Block = 16 tokens, code levels [4, 8] (codebook 1024), prefix 64, 100000 training windows (50% are the original model's own answers to chat prompts; 100% of the rest distilled from it).
Generator backbone: **llm** (full attention), 50312 steps, batch 32.
Passes per block = 2 code passes + R token passes (+1 to cache the finished block); the original model needs 16 passes per block.

## Quality and speed vs the original model

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.518 | 1.0 | 0.568 | 0.078 | 29.5 | 1.016 | 11/40 |
| NCM refine=1 | 5.689 | 0.122 | 0.52 | 0.032 | 65.5 | 0.25 | - |
| NCM refine=2 | 5.521 | 0.115 | 0.547 | 0.058 | 51.9 | 0.313 | - |
| NCM refine=4 | 5.537 | 0.112 | 0.57 | 0.051 | 37.2 | 0.44 | 0/40 |
| real held-out text (reference) | 2.33 | - | - | - | - | - | - |

Judge bits/token: the original model's surprise at each output, lower = closer to what it would write. The original's own greedy output is the floor; real held-out text shows what normal human-written text scores.

## Compressor

Token reconstruction accuracy from codes alone (held-out): 1 level(s): 20.2%, 2 level(s): 47.1%; codebook use per level: [0.798, 0.781].

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
 get___info(sock____):):
    if is_is_sock:
       
        sock sock:
       
        print("Errorsock: + " "")
        if not.is:
       
       .: "Error""
        except:
            print(" "Errorsock: " " "
```

**NCM refine=2**

```
 get_check_func(sock,sock_str):):
    if is_check_sock):
        return True
    else:
        sock[sock0] = sock + + return[

def get_checkcheckfunc(sock,sock_str, check_str, "check):
    if if
```

**NCM refine=4**

```
 get_check_func(sock_sock_server):):
    if is_check_server:
        return True
    else:
        sock_check_func = ' get_check_server
   def is_check_:
    if if notsockcheckfunc:
        return is_check_func_func
```

