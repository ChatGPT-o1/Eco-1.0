# llm2ncm-draft: draft heads for HuggingFaceTB/SmolLM2-135M-Instruct

Every speculative output is checked token-for-token against plain greedy decoding ('exact match').

## eagle-ttt4: 4-step training-time test (EAGLE-3 style) (exit 0)

### float32 (56 prompts, up to 192 new tokens; head from step 6000)

| decoder | exact match | accepted drafts / round | tokens per pass | tokens/s | speed-up vs plain | vs HF generate |
|---|---|---|---|---|---|---|
| plain greedy (1 token per pass) | - | - | 1.00 | 31.3 | 1.00x | 1.01x |
| draft head, k=3 | 56/56 | 0.67 of 3 | 1.65 | 41.4 | **1.32x** | 1.34x |
| draft head, k=5 | 56/56 | 0.71 of 5 | 1.69 | 38.2 | **1.22x** | 1.23x |

### float16 (56 prompts, up to 192 new tokens; head from step 6000)

| decoder | exact match | accepted drafts / round | tokens per pass | tokens/s | speed-up vs plain | vs HF generate |
|---|---|---|---|---|---|---|
| plain greedy (1 token per pass) | - | - | 1.00 | 26.4 | 1.00x | 1.00x |
| draft head, k=3 | 48/56 | 0.67 of 3 | 1.65 | 37.5 | **1.42x** | 1.43x |
| draft head, k=5 | 48/56 | 0.71 of 5 | 1.70 | 34.9 | **1.32x** | 1.33x |

## eagle-ttt1: one-step training (EAGLE-1 style), control (exit 0)

### float32 (56 prompts, up to 192 new tokens; head from step 11899)

| decoder | exact match | accepted drafts / round | tokens per pass | tokens/s | speed-up vs plain | vs HF generate |
|---|---|---|---|---|---|---|
| plain greedy (1 token per pass) | - | - | 1.00 | 30.9 | 1.00x | 1.00x |
| draft head, k=3 | 56/56 | 0.80 of 3 | 1.78 | 44.1 | **1.43x** | 1.43x |
| draft head, k=5 | 56/56 | 0.84 of 5 | 1.82 | 40.4 | **1.31x** | 1.31x |

### float16 (56 prompts, up to 192 new tokens; head from step 11899)

| decoder | exact match | accepted drafts / round | tokens per pass | tokens/s | speed-up vs plain | vs HF generate |
|---|---|---|---|---|---|---|
| plain greedy (1 token per pass) | - | - | 1.00 | 26.2 | 1.00x | 1.00x |
| draft head, k=3 | 48/56 | 0.81 of 3 | 1.79 | 40.5 | **1.54x** | 1.55x |
| draft head, k=5 | 48/56 | 0.84 of 5 | 1.82 | 37.5 | **1.43x** | 1.43x |

