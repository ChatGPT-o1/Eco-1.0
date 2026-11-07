# V14: replication guide

V14 is a careful form of reinforcement learning that makes **SmolLM2-135M-Instruct** better at writing short Python
functions that pass unit tests (MBPP), while keeping its ordinary chat behaviour almost unchanged. This folder holds
everything needed to rebuild and rerun it from scratch: the code, the two small data files it needs, a run script, and
this guide.

## What you get

```
v14/
  REPLICATE.md                     this guide
  run_v14.sh                       train + score, resumable
  crl/
    smol_code.py                   MBPP loading, sandboxed test runner
    evaluate.py                    HumanEval helpers (prompt, program builder, runner)
    v8/
      engine.py                    fast answer writer, grading, checkpoints (CPU or GPU)
      train.py                     the trainer: --method v14 (also grpo = plain RL, and older v8/v9/v10)
      bank.py                      "screen" stage: how often the model solves each training problem (makes data/screen.json)
      evalx.py                     final test: 500 MBPP test + 164 HumanEval
      compare.py                   problem-by-problem comparison with 95% bootstrap intervals
      data/screen.json             screening result for SmolLM2-135M (8 answers per MBPP training problem)
  scr/local/msdata/heldout_prompts.jsonl   ordinary chat prompts, used to keep chat unchanged
```

## 1. Setup

- Python 3.12 or newer (tested with 3.13), 4 CPU cores, about 9 GB of RAM, about 4 GB of free disk.
- Network access to the Hugging Face Hub (the model and the MBPP / HumanEval datasets are downloaded on first use).

```bash
pip install torch transformers==5.19.0 huggingface_hub pyarrow
```

`transformers` 5.19 matters: the answer writer uses `DynamicCache.batch_repeat_interleave` / `batch_select_indices`,
`logits_to_keep`, and the `dtype=` argument of `from_pretrained`.

Environment variables (all optional):

| variable | default | meaning |
|---|---|---|
| `ENGINE_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` | starting model (hub id or local folder) |
| `ENGINE_SCR` | `v14/scr` | folder holding `local/msdata/heldout_prompts.jsonl` |
| `ENGINE_DEVICE` | `cpu` | `cuda` runs everything on the GPU in float32 (see section 7) |
| `HF_HOME` | Hugging Face default | download cache |

## 2. Check the install (1 minute)

```bash
python3 -I v14/crl/v8/train.py --selfcheck
```

Expected:

```
(1) V8 vs GRPO gradient rel err 0.00e+00 -> PASS
(2) half-gradients sum vs full rel err 1.05e-06 -> PASS
(3) preserve KL at base 0.00e+00, after 1% noise 0.005 -> PASS
(4) bf16 vs fp32 log-prob gap: mean |d| 0.015, k3 noise floor 5e-04 (vs --eps-task 0.003)
```

## 3. Train and score

```bash
./v14/run_v14.sh 120
```

This trains for 120 compute-minutes (time is summed across restarts; a crashed or killed run resumes from its last
checkpoint when you run the script again), then scores the starting model and V14 on all 500 MBPP test problems and
all 164 HumanEval problems, and writes `v14/evals/compare.txt`.

The training command on its own:

```bash
cd v14/crl/v8
python3 -I train.py --method v14 --out ../../runs/v14 --budget-min 120 --threads 4 --micro 4 --no-teacher 1
```

`--no-teacher 1` makes it pure RL: problems the model always fails get a known-good answer only from the model's own
earlier successes. Without it, V14 also uses `data/teacher.json` if that file exists (a map from task id to a list of
correct solutions; `bank.py teacher` makes one with Qwen2.5-Coder-0.5B). The recorded results below used no teacher.

The plain-RL baseline (GRPO) with the same split and checks, for a fair comparison:

```bash
python3 -I train.py --method grpo --pick-split 1 --out ../../runs/grpo --budget-min 120 --threads 4 --micro 4
```

### Outputs

| file | content |
|---|---|
| `runs/v14/train_log.jsonl` | one JSON line per update (pass rates, step sizes, KL values, held-out scores) |
| `runs/v14.log` | human-readable log, one line per update |
| `runs/v14/ckpt_a.pt`, `ckpt_b.pt`, `latest.json` | alternating checkpoints (weights + optimizer averages in bf16), every 5 minutes |
| `runs/v14/best.pt` | best weights on the held-out problems |
| `runs/v14/final/` | the final model (best weights, then the end jump), loadable with `from_pretrained` |
| `runs/v14/final.json` | what was chosen and why |
| `evals/*.json`, `evals/compare.txt` | final test results |

A typical log line:

```
update 20 | store 30 groups + 4 known-good | update 56s | rho 0.013 step kept s 3.6e-05 kl_step 0.0044
kl_stale 0.0121 pres 0.0305 | compute 32.3/120.0 min | peak mem 8.3 GB | DEV 0.108 ± 0.018 best
```

`DEV` is the held-out score. `pres` is the chat change (KL to the starting model on ordinary chat replies).

## 4. How V14 works

**Data split.** MBPP's 374 training problems are shuffled with `random.Random(1234)`. The last 74 are held out and used
only for choices: which checkpoint to keep and how far to jump at the end. V14 trains on the other 300. The 500 MBPP
test problems and HumanEval are touched only by the final test.

**Answer writer** (`engine.sample`). A hand-written decode loop with a KV cache. Each prompt is pre-filled once and the
cache copied for its answers. A row stops at an end token, at 160 new tokens, or as soon as its text contains a closed
` ```python ` block. The grader only reads the first closed block, so nothing after it matters. Finished rows are dropped
from the batch.

**Grading.** An answer scores 1 only if the extracted function passes every MBPP test for the problem. Tests run in a
separate Python process with a 2 GB memory limit and a 3 s timeout. There is no partial credit: `return False` would
otherwise pass some tests.

**Which problems to practise.** `data/screen.json` holds how many of 8 answers passed for each training problem. A
beta-binomial prior is fitted to those counts; most problems are never solved, so the prior is skewed towards 0. Each
problem's weight is the posterior probability that a group of 6 answers is mixed (some pass, some fail). Only mixed
groups carry an RL signal. With this prior, about 30% of draws go to problems the model has never solved; with a flat
prior it was about 75%.

**Rolling answer store.**
1. Whenever the store holds fewer than 6 useful groups, a quarter of the groups have been used 3 times, or 2 updates in
   a row failed, it writes a chunk.
2. A chunk covers 32 problems. Stage 1 writes 2 answers per problem at T=1. Stage 2 writes 6 more only where stage 1
   had a pass, or where the sampler still expects mixed results (weight ≥ 0.3).
3. Each mixed group gets advantages `(r - mean) / (std + 1e-4)`, and the training copy of the model scores its
   answers' log-probabilities once.
4. All-fail problems get a known-good answer if one exists: one of the model's own earlier passing answers, or a
   teacher solution.
5. The store keeps at most 30 groups (the most-used are dropped) and 16 known-good answers.

**One update.**
1. Take the 6 least-used groups (48 answers) and 2 known-good answers.
2. The objective is the PPO-clipped surrogate over the answers, plus 0.2 × the mean log-likelihood of the known-good
   answers. In it, `ratio = exp(logp - logp_when_scored)` and clip = 0.2.
3. Compute the gradient twice, once per half of the groups. The halves' cosine is logged as `rho`.
4. Update Adam-style averages: m with β1 = 0.9, v with β2 = 0.99.
5. Try a step on the real model, in full precision:
   `W = W_here - s * m_hat / (sqrt(v_hat) + 1e-8)`, starting at `s = min(s_max, 2 * s_prev)`.
6. Keep the step only if all of these hold:
   - the objective went up;
   - KL(current → new) on the minibatch is at most 0.015;
   - KL(writer → new) is at most 0.08, so stored answers have not gone stale;
   - chat change is at most 0.1. Chat change is KL(start → new) on 24 ordinary chat replies, with the starting model's
     distribution coarsened to its top 512 tokens plus the rest.
7. If a KL check fails, shrink s by `max(0.125, 0.9 * sqrt(limit / measured))`, capped at ½. Otherwise halve it. Up to 4
   tries.
8. After a kept step, `s_prev = s * clamp(0.5 * sqrt(limit / measured), 0.25, 1)`. After a rejected round,
   `s_prev = 0.25 * last kept s`, the gentle recovery.

**Checks and choices.**
- **Held-out check:** every 10 updates, V14 scores the 74 held-out problems (4 answers each, T=0.8, a fresh seed
  `999 + update` each time). It saves `best.pt` when the score improves, and rolls back to the best weights if the
  score falls more than 2 combined standard errors below the best.
- **End jump:** at the end, V14 loads the best weights and tries `W_start + k * (W_best - W_start)` for k in
  {1, 1.5, 2, 3, 5}. Each candidate is scored on the held-out problems with 8 answers each and one shared seed. The best
  candidate whose chat change is at most 0.05 is kept.

**Main settings** (V14 defaults in `train.py`):

| setting | value | setting | value |
|---|---|---|---|
| `--eps-task` (KL per update) | 0.015 | `--eps-pool` (staleness cap) | 0.08 |
| `--eps-pres` (chat cap) | 0.1 | `--s-init` / `--s-max` / `--s-min` | 5e-6 / 3e-4 / 1e-7 |
| `--tries` | 4 | `--gentle-recovery` | on |
| `--chunk-prompts` / `--pool-group` / `--stage1` | 32 / 8 / 2 | `--pool-groups` / `--reuse` | 30 / 3 |
| `--mb-groups` / `--mb-rescue` | 6 / 2 | `--rescue-weight` | 0.2 |
| `--clip` | 0.2 | `--beta1` / `--beta2` | 0.9 / 0.99 |
| `--temperature` / `--max-new` | 1.0 / 160 | `--pick-size` | 74 |
| `--dev-every` / `--dev-n` | 10 / 4 | `--end-jumps` / `--jump-n` / `--jump-chat-cap` | 1.5,2,3,5 / 8 / 0.05 |

## 5. Recorded results

These are from one run, on a 4-core CPU (Intel Xeon, AMX).

| measure | value |
|---|---|
| held-out (74 problems) at start | 7.1% |
| held-out after update 10 (16 min) / update 20 (32 min) | 9.1% / **10.8%** |
| updates in 37 min | 24 (18 kept, 6 rejected by the checks) |
| time per update | 45–90 s; writing a chunk of about 100–160 answers takes 85–95 s |
| chat change after 32 min | 0.03 |
| MBPP test (500 problems, 2 sampled answers each, T=0.8), update-20 weights | **7.9%** (starting model about 5.7%, measured with 160-token answers) |
| the same weights plus the example-test retry (up to 4 tries) | **14.6%** |

The 120-minute run was stopped at 37 minutes, so the end jump and the full HumanEval test have not been run on V14 yet.
One random seed only; gains of 1–2 points are within the noise of a 74-problem check.

## 6. Pitfalls already fixed (do not reintroduce)

1. **Step direction.** `.grad` holds the gradient of the negative objective, so the step subtracts:
   `W - s * m_hat / ...`. An earlier version added it, and every candidate step made the objective worse.
2. **Full-precision checks.** Weight changes of this size (about 1e-6 relative) are below bf16 resolution. The
   objective and KL checks must use float32 forward passes, or they measure rounding noise: about 8e-4 of KL at every
   step size. Gradients can use bf16 autocast (`--autocast 1`).
3. **Stochastic rounding for the bf16 writer copy.** On CPU, the copy that writes answers is bf16. It is refreshed with
   unbiased (stochastic) rounding so small updates reach it on average.
4. **The sampler prior.** A flat Beta(1, 1) prior sends about 75% of draws to never-solved problems. Fit the prior to
   the screening counts.
5. **Memory.** Peak is about 8 GB. Keep `--micro 4` on a 15 GB machine shared with other jobs.
6. **Noisy choices.** Picking checkpoints and jump sizes on the same problems with the same seed inflates scores, by
   about +6 points in one test. Hence the separate held-out set, fresh seeds per check, and the untouched test set.
7. **Chat drift from big jumps.** Jumping multiplies all accumulated change, including noise, so the end jump is capped
   at chat change 0.05.

## 7. GPU

Set `ENGINE_DEVICE=cuda`. Everything then runs in float32 on the GPU (a T4 has no fast bf16). Pass `--autocast 0`, and
use `CUDA_VISIBLE_DEVICES` to pin one run per GPU. This path passes the CPU self-checks, but a full GPU run had not
finished when this guide was written.

## 8. Making it work for another model

1. Set `ENGINE_MODEL`.
2. Re-run the screening:
   `python3 -I bank.py screen --out data` (8 answers per training problem; about 18 minutes on CPU for SmolLM2).
3. Check the self-tests.
4. If the model's answers do not use a ` ```python ` block, adapt `engine.CLOSED` and `extract_code` in `evaluate.py`.
