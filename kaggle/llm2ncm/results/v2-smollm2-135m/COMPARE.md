# llm2ncm-v2: variants compared

Reference (earlier run): v1 (50,000 windows, original recipe): NCM refine=4 judge 4.768 bits/token, agreement 0.060, distinct-2 0.23, HumanEval 0/20; the original model: 0.546 / 1.0 / 0.688 / 2/20

## v2: v2 data + per-slot embeddings + compressor LR decay (codes 2+8)

`` (exit 0)

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.546 | 1.0 | 0.688 | 0.076 | 29.2 | 1.016 | 2/20 |
| NCM refine=1 | 4.559 | 0.087 | 0.27 | 0.0 | 125.7 | 0.127 | - |
| NCM refine=2 | 5.086 | 0.098 | 0.334 | 0.059 | 100.0 | 0.158 | - |
| NCM refine=4 | 5.237 | 0.084 | 0.384 | 0.032 | 73.9 | 0.221 | 0/20 |
| real held-out text (reference) | 2.832 | - | - | - | - | - | - |


Generated on Kaggle with `ncm_generate.py --refine 4 --new 128`.

## What is 2 + 2?

```
To solve this  2 2, we can use the basic of arithmetic. and. the addition. the addition.. the.. addition.


The,, we we we,, we we we we add the addition.



.




.


...


..


..



..




...



...

.....
..
.
......
....
.
...
```

[128 tokens, 28 forward passes, 59.4 tok/s]

## What is the capital of France?

```
The capital of France France is is the in of of of France..
```

[15 tokens, 7 forward passes, 18.3 tok/s]

## Write a Python function that adds two numbers.

```
Here's a Python function that adds two numbers of two numbers:

```pythonpython
defdef add__numbers(numbers,,,,,)
    return =(numbers(numbers,numbers,,(,(numbers))
This function takesates the numbers of numbers of the the of of of,,, returns the numbers of the of of the numbers of of of of of of. of of of numbers numbers of of the of.. of ofnumbers of of of of of of of numbers of of of numbers of the of of of of of of of of of of of of numbers of of of of
```

[128 tokens, 28 forward passes, 57.9 tok/s]


## v2-wide: same, with twice the code slots (4+16): 2 tokens per fine code instead of 4

`--scales 4,16` (exit 0)

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.546 | 1.0 | 0.688 | 0.076 | 29.8 | 1.016 | 2/20 |
| NCM refine=1 | 5.065 | 0.07 | 0.367 | 0.075 | 131.1 | 0.125 | - |
| NCM refine=2 | 4.832 | 0.074 | 0.408 | 0.176 | 107.7 | 0.156 | - |
| NCM refine=4 | 4.92 | 0.083 | 0.439 | 0.03 | 75.3 | 0.219 | 0/20 |
| real held-out text (reference) | 2.832 | - | - | - | - | - | - |


Generated on Kaggle with `ncm_generate.py --refine 4 --new 128`.

## What is 2 + 2?

```
To solve  +  2,2,2, ,, we can use use the operations..
```

[23 tokens, 7 forward passes, 28.8 tok/s]

## What is the capital of France?

```
The capital is the the is is the City,,,, Paris- Paris- Paris- Paris, Paris, Paris-s, Paris, Paris,, Paris,- Paris- Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris,, Paris, Paris,, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris,,, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris, Paris,,,,, Paris, Paris
```

[128 tokens, 28 forward passes, 60.7 tok/s]

## Write a Python function that adds two numbers.

```
Here's a Python function that adds adds two two:

```python
def_____numbers(a, num,,,,,): return return
   
    return
   ,a,
    return num

This function takes takes two two two numbers and and, num, as as as as arguments. It returns them and function and and, returns, returns, as num, and,,,, and, as num as as as as as as arguments.


NoteThis function function takes two two two, and and, and and and as as as as as as as as as as as as
```

[128 tokens, 28 forward passes, 63.8 tok/s]


