# llm2ncm-final: variants compared

Reference (earlier run): NCM refine=4 (judge bits/token / agreement / distinct-2 / HumanEval): v1 (50k windows, original recipe) 4.768 / 0.060 / 0.23 / 0/20; v2 (VQ, 2+8 slots) 5.237 / 0.084 / 0.384 / 0/20; v2-wide (VQ, 4+16 slots) 4.920 / 0.083 / 0.439 / 0/20; the original model 0.546 / 1.0 / 0.688 / 2/20. Compressor token recovery from codes (1 level / 2 levels): v1 9.6% / 20.4%, v2 11.3% / 22.5%, v2-wide 13.5% / 43.1%.; v3 (FSQ, 2+8) 5.331 / 0.097 / 0.351 / 0/20; v3-wide (FSQ, 4+16) 4.425 / 0.078 / 0.366 / 0/20

## final-b32: 32-word blocks, 4+16 code slots

`--scales 4,16` (exit 0)

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.518 | 1.0 | 0.568 | 0.078 | 29.4 | 1.016 | 11/40 |
| NCM refine=1 | 5.89 | 0.118 | 0.502 | 0.024 | 129.2 | 0.125 | - |
| NCM refine=2 | 5.695 | 0.12 | 0.525 | 0.03 | 103.6 | 0.156 | - |
| NCM refine=4 | 5.462 | 0.114 | 0.533 | 0.047 | 73.6 | 0.219 | 0/40 |
| real held-out text (reference) | 2.33 | - | - | - | - | - | - |


Generated on Kaggle with `ncm_generate.py --refine 4 --new 128`.

## What is 2 + 2?

```
The answer 22, is a simple and elegant operation. It is the the the first two two after together. We two two different numbers.

The sum 2 +2 is: 2 2 2.
```

[47 tokens, 14 forward passes, 36.4 tok/s]

## What is the capital of France?

```
The capital of France is Paris.
```

[7 tokens, 7 forward passes, 8.4 tok/s]

## Write a Python function that adds two numbers.

```
Here's a Python function that adds two numbers:

```python
def add__(numbers):
    return return
    a + b + b + b1

# Example: usage::
numbers =1 =1,2,2, 2,3 +2, 3 + 3, 4,4, 5,5)
numbers
numbers)  Output: Output:1,1,
```

[92 tokens, 21 forward passes, 54.5 tok/s]


## final-b16: 16-word blocks, 4+8 code slots

`--scales 4,8` (exit 0)

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.518 | 1.0 | 0.568 | 0.078 | 29.5 | 1.016 | 11/40 |
| NCM refine=1 | 5.689 | 0.122 | 0.52 | 0.032 | 65.5 | 0.25 | - |
| NCM refine=2 | 5.521 | 0.115 | 0.547 | 0.058 | 51.9 | 0.313 | - |
| NCM refine=4 | 5.537 | 0.112 | 0.57 | 0.051 | 37.2 | 0.44 | 0/40 |
| real held-out text (reference) | 2.33 | - | - | - | - | - | - |


Generated on Kaggle with `ncm_generate.py --refine 4 --new 128`.

## What is 2 + 2?

```
22 + 22, we use the concept of algebraic notation that states that the adding two numbers, a number (a b) and the term is understood to add a a quantity.

In this case, a2 is the value of the quantity, which is a way representation that that we add the two quantities into a quantity.
2 + 2) is a quantity of quantity, that is 2 + 2, and a quantity value of quantity, quantity is 22, 2 2, the value value of 2 + 2 2, The value22 the value of value
```

[128 tokens, 56 forward passes, 34.5 tok/s]

## What is the capital of France?

```
The capital of France is Paris.
```

[7 tokens, 7 forward passes, 8.4 tok/s]

## Write a Python function that adds two numbers.

```
Here's a Python function that adds two numbers:

```
def add_numbers(a, b):
    return + a + a2)
        | b + (21
       22222) and the result of 2)
```

# Example usage:
add_numbers(numbers(1 32
add_numbers(2,2)
 # Output: 1
addadd_numbers(numbers(1 322
# Output Example Example:
```

ThisThis function adds two numbers using two numbers` and ` `a`. The ` `
```

[128 tokens, 56 forward passes, 33.6 tok/s]


