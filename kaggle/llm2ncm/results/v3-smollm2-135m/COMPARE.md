# llm2ncm-v3: variants compared

Reference (earlier run): NCM refine=4 (judge bits/token / agreement / distinct-2 / HumanEval): v1 (50k windows, original recipe) 4.768 / 0.060 / 0.23 / 0/20; v2 (VQ, 2+8 slots) 5.237 / 0.084 / 0.384 / 0/20; v2-wide (VQ, 4+16 slots) 4.920 / 0.083 / 0.439 / 0/20; the original model 0.546 / 1.0 / 0.688 / 2/20. Compressor token recovery from codes (1 level / 2 levels): v1 9.6% / 20.4%, v2 11.3% / 22.5%, v2-wide 13.5% / 43.1%.

## v3: FSQ + predictability loss + autoregressive outline + soft distillation (codes 2+8)

`--comp-extra=--quant fsq --fsq-levels 8,5,5,5 --pred-weight 0.1 --lr-final 0.1 --gen-extra=--slot-emb --ar-first-level --soft-weight 0.5` (exit 0)

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.546 | 1.0 | 0.688 | 0.076 | 29.8 | 1.016 | 2/20 |
| NCM refine=1 | 4.73 | 0.113 | 0.226 | 0.049 | 106.5 | 0.156 | - |
| NCM refine=2 | 5.008 | 0.107 | 0.312 | 0.076 | 84.8 | 0.188 | - |
| NCM refine=4 | 5.331 | 0.097 | 0.351 | 0.108 | 66.2 | 0.25 | 0/20 |
| real held-out text (reference) | 2.832 | - | - | - | - | - | - |


Generated on Kaggle with `ncm_generate.py --refine 4 --new 128`.

## What is 2 + 2?

```
To find the number2   222,,
```

[13 tokens, 8 forward passes, 13.1 tok/s]

## What is the capital of France?

```
The capital of is is a country country.
```

[9 tokens, 8 forward passes, 9.0 tok/s]

## Write a Python function that adds two numbers.

```
Here's a Python function that adds two two two numbers numbers:

python
add_(num):
    num num num::
   
   ::
```

This function uses a two arguments, which to to the the result and to the result and and then to the the result.
```

Example example

```


```python

```

```


```



```


```

```


```


```
```
``````````````````

``````

`````````````````````
```

[128 tokens, 32 forward passes, 51.4 tok/s]


## v3-wide: same, with twice the code slots (4+16)

`--scales 4,16 --comp-extra=--quant fsq --fsq-levels 8,5,5,5 --pred-weight 0.1 --lr-final 0.1 --gen-extra=--slot-emb --ar-first-level --soft-weight 0.5` (exit 0)

| system | judge bits/token ↓ | agreement w/ original | distinct-2 | repeated lines | tokens/s | passes/token | HumanEval |
|---|---|---|---|---|---|---|---|
| original (token-by-token) | 0.546 | 1.0 | 0.688 | 0.076 | 30.4 | 1.016 | 2/20 |
| NCM refine=1 | 4.436 | 0.101 | 0.283 | 0.022 | 78.5 | 0.219 | - |
| NCM refine=2 | 4.796 | 0.092 | 0.375 | 0.058 | 67.7 | 0.25 | - |
| NCM refine=4 | 4.425 | 0.078 | 0.366 | 0.018 | 55.0 | 0.312 | 0/20 |
| real held-out text (reference) | 2.832 | - | - | - | - | - | - |


Generated on Kaggle with `ncm_generate.py --refine 4 --new 128`.

## What is 2 + 2?

```
To the is 2 +2222 =2222222222.
```

[22 tokens, 10 forward passes, 20.1 tok/s]

## What is the capital of France?

```
The capital of is Paris is a city and a France, located the heart of the heart of the of the of the famous of of the city. Paris is a vibrant metropolis,, center of the history of culture of France of the city of the city of of France of the city of the of city of the of France, of the city.
```

[70 tokens, 30 forward passes, 31.2 tok/s]

## Write a Python function that adds two numbers.

```
Here's a Python function that adds two two two numbers using a input:

```python

def addnumbers((a, num, num, b):
    return a,, num, num, num, num, num, num, num)

# Example a num, num, num, num, num,  11, 1, 2, , , num)
print,,, num, num, num, num, num, num,, num, num num, num, num, 11, num, , num, num, 12,
```

[128 tokens, 40 forward passes, 45.2 tok/s]


