# GrandeRL-1.0

GrandeRL-1.0 is a careful form of reinforcement learning, developed in this project as "V14". It learns from the
model's own graded answers, like ordinary RL (GRPO), and it:

- gets more learning out of every answer it writes;
- tries each weight change on the real model before keeping it;
- caps how much the model can change;
- protects ordinary chat behaviour;
- rolls back if it slips;
- needs no learning rate.

It is set up here for **SmolLM2-135M-Instruct** writing short Python functions that must pass unit tests (MBPP).

## Results so far

Two 10-minute races on Kaggle T4 GPUs, one GPU per method. Both methods started from the same model, trained on the
same 300 problems, chose checkpoints on the same 74 held-out problems and got no outside help. They were then scored
on 500 unseen MBPP test problems and the 164 HumanEval problems.

| MBPP test (500) | Race 1 | Race 2 (other seed) |
|---|---|---|
| starting model | 6.4% | 4.8% |
| plain RL (GRPO) | 6.7% | 6.4% |
| **GrandeRL-1.0** | **8.2%** | **7.0%** |
| GrandeRL − plain RL | +1.5 (95% CI −0.2 to +3.3) | +0.6 (95% CI −1.1 to +2.3) |
| GrandeRL − start | +1.8 (95% CI +0.1 to +3.6) | +2.2 (95% CI +0.4 to +4.1) |

| HumanEval (164, 10 answers per problem) | Race 1 (re-scored) | Race 2 |
|---|---|---|
| plain RL (GRPO) | 7.32% | 7.87% |
| **GrandeRL-1.0** | 7.26% | 7.01% |
| GrandeRL − plain RL | −0.06 (95% CI −1.0 to +0.9) | −0.85 (95% CI −1.8 to +0.2) |

**Summary:**
- GrandeRL-1.0 beat plain RL on MBPP, the kind of problem it practised, in both races, by about 1 point on average.
  Each margin alone is within the noise, so more seeds are needed to call it proven.
- It clearly improved on the starting model.
- HumanEval is a tie.
- In race 2, plain RL peaked early (held-out 9.8% at round 5) and then drifted down to 7.1%. GrandeRL kept improving,
  to 12.8% at update 15.

An earlier 37-minute CPU run reached 10.8% on the held-out problems (7.1% at the start) and 7.9% on the MBPP test.
With an answer-time example-test retry (up to 4 tries, keeping the first answer that passes the example test shown in
the problem), the same weights reached 14.6%.

## What is in this folder

```
GrandeRL-1.0/
  README.md                          this guide
  run_granderl.sh                    train + score on CPU, resumable
  kaggle/build_race.py               builds a Kaggle GPU kernel: plain RL vs GrandeRL side by side, then the full test
  crl/
    smol_code.py                     MBPP loading, sandboxed test runner
    evaluate.py                      HumanEval helpers
    v8/
      engine.py                      fast answer writer, grading, checkpoints (CPU or GPU)
      train.py                       the trainer: --method granderl (also grpo = plain RL)
      bank.py                        screening: how often the model solves each training problem
      evalx.py                       final test: 500 MBPP test + 164 HumanEval
      compare.py                     problem-by-problem comparison with 95% bootstrap intervals
      data/screen.json               screening result for SmolLM2-135M
  scr/local/msdata/heldout_prompts.jsonl   ordinary chat prompts (used to keep chat unchanged)
```

## 1. Setup

- Python 3.12 or newer (tested on 3.13).
- On CPU: 4 cores, about 9 GB RAM and 4 GB disk. A GPU also works (section 6).
- Network access to the Hugging Face Hub; the model and datasets download on first use.

```bash
pip install torch transformers==5.19.0 huggingface_hub pyarrow
```

`transformers` 5.19 matters: the answer writer uses `DynamicCache.batch_repeat_interleave`, `batch_select_indices`
and `logits_to_keep`, and the `dtype=` argument of `from_pretrained`.

| variable | default | meaning |
|---|---|---|
| `ENGINE_MODEL` | `HuggingFaceTB/SmolLM2-135M-Instruct` | starting model (hub id or local folder) |
| `ENGINE_SCR` | `GrandeRL-1.0/scr` | folder holding `local/msdata/heldout_prompts.jsonl` |
| `ENGINE_DEVICE` | `cpu` | `cuda` runs everything on the GPU in float32 |
| `HF_HOME` | Hugging Face default | download cache |

## 2. Check the install (1 minute)

```bash
python3 -I GrandeRL-1.0/crl/v8/train.py --selfcheck
```

Expected: checks (1) to (3) print `PASS`, and (4) reports a k3 noise floor of about 5e-04.

## 3. Train and score on CPU

```bash
./GrandeRL-1.0/run_granderl.sh 120
```

This trains for 120 compute-minutes. Time is summed across restarts: running the script again resumes from the last
checkpoint. It then scores the starting model and GrandeRL on 500 MBPP test and 164 HumanEval problems, and writes
`GrandeRL-1.0/evals/compare.txt`.

To train directly:

```bash
cd GrandeRL-1.0/crl/v8
python3 -I train.py --method granderl --out ../../runs/granderl --budget-min 120 --threads 4 --micro 4 --no-teacher 1
```

The plain-RL baseline, with the same split and checks:

```bash
python3 -I train.py --method grpo --pick-split 1 --out ../../runs/grpo --budget-min 120 --threads 4 --micro 4
```

`--no-teacher 1` keeps it pure RL: known-good answers for problems the model keeps failing come only from its own
earlier successes. Without it, GrandeRL also uses `data/teacher.json` if that file exists. The file maps each task id
to a list of correct solutions; `bank.py teacher` makes one with Qwen2.5-Coder-0.5B.

### Outputs

| file | content |
|---|---|
| `runs/granderl.log` | one line per update; `DEV` is the held-out score, `pres` the chat change |
| `runs/granderl/train_log.jsonl` | the same, as JSON |
| `runs/granderl/ckpt_a.pt`, `ckpt_b.pt`, `latest.json` | alternating checkpoints, every 5 minutes |
| `runs/granderl/best.pt` | best weights on the held-out problems |
| `runs/granderl/final/` | final model (best weights, then the end jump), loadable with `from_pretrained` |
| `evals/*.json`, `evals/compare.txt` | final test results |

## 4. Race it against plain RL on Kaggle (2 T4 GPUs)

```bash
python3 GrandeRL-1.0/kaggle/build_race.py ten race_ten YOUR_KAGGLE_USERNAME
kaggle kernels push -p race_ten --accelerator NvidiaTeslaT4 -t 5400
kaggle kernels status YOUR_KAGGLE_USERNAME/granderl-vs-grpo-ten
kaggle kernels output YOUR_KAGGLE_USERNAME/granderl-vs-grpo-ten -p race_ten_out
```

The stages are:

| stage | training | test | total time |
|---|---|---|---|
| `smoke` | 4 min per method | 12-problem test | about 15 min |
| `ten` | 10 min per method | full test, 2 answers per problem | about 40 min |
| `seed1` | as `ten`, with another training seed | HumanEval scored with 10 answers per problem | about 60 min |
| `full` | 120 min per method | 5 answers per problem | |

The kernel downloads everything itself and writes `compare.txt` plus all logs to its output. It scores the starting
model, plain RL, GrandeRL without its end jump (the same-compute comparison), and GrandeRL with it.

## 5. How it works

**Data split.** MBPP's 374 training problems are shuffled with `random.Random(1234)`. The last 74 are held out for
choices only: which checkpoint to keep, and how far to jump at the end. GrandeRL trains on the other 300. The test
sets are touched only by the final test, and every held-out check uses a fresh sampling seed.

**Answer writer** (`engine.sample`).
- A hand-written decode loop with a KV cache. Each prompt is pre-filled once and the cache copied for its answers.
- A row stops at an end token, at 160 new tokens, or as soon as its text contains a closed ` ```python ` block. The
  grader only reads the first closed block.
- Finished rows are dropped from the batch.

**Grading.** An answer scores 1 only if the extracted function passes every MBPP test. Tests run in a separate
process with a 2 GB memory limit and a 3 s timeout. There is no partial credit: `return False` would pass some tests.

**Which problems to practise.**
- `data/screen.json` holds how many of 8 answers passed for each training problem.
- A beta-binomial prior is fitted to those counts. It is skewed towards 0, because most problems are never solved.
- Each problem's weight is the posterior probability that a group of 6 answers is mixed (some pass, some fail). Only
  mixed groups carry an RL signal.
- About 30% of draws go to never-solved problems; with a flat prior it was about 75%.

**Rolling answer store.**
- A new chunk is written whenever the store holds fewer than 6 useful groups, a quarter of the groups have been used
  3 times, or 2 updates in a row failed.
- A chunk covers 32 problems. Stage 1 writes 2 answers each at T=1. Stage 2 writes 6 more only where stage 1 had a
  pass, or where the sampler weight is at least 0.3.
- Each mixed group gets advantages `(r - mean) / (std + 1e-4)`, and the training copy scores its answers'
  log-probabilities once.
- An all-fail problem gets a known-good answer if one exists: an earlier passing answer of its own, or a teacher
  solution.
- The store keeps at most 30 groups (the most-used are dropped) and 16 known-good answers.

**One update.**
1. Take the 6 least-used groups (48 answers) and 2 known-good answers.
2. The objective is the PPO-clipped surrogate (clip 0.2, `ratio = exp(logp - logp_when_scored)`) plus 0.2 × the mean
   log-likelihood of the known-good answers.
3. Compute the gradient once per half of the groups. The halves' cosine is logged as `rho`.
4. Update Adam-style averages: m with β1 = 0.9, v with β2 = 0.99.
5. Try `W = W_here - s * m_hat / (sqrt(v_hat) + 1e-8)` on the real model in float32, starting at
   `s = min(3e-4, 2 * s_prev)`.
6. Keep the step only if all of these hold:
   - the objective went up;
   - KL(current → new) on the minibatch is at most 0.015;
   - KL(writer → new) is at most 0.08, so stored answers have not gone stale;
   - chat change is at most 0.1. Chat change is KL(start → new) on 24 ordinary chat replies, with the starting
     distribution coarsened to its top 512 tokens plus the rest.
7. If a KL check fails, shrink s by `max(0.125, 0.9 * sqrt(limit / measured))`, capped at ½. Otherwise halve it. Up to
   4 tries.
8. After a kept step, `s_prev = s * clamp(0.5 * sqrt(limit / measured), 0.25, 1)`. After a rejected update,
   `s_prev = 0.25 * last kept s`.

**Checks and the end jump.**
- **Held-out check:** every 10 updates, score the 74 held-out problems (4 answers each, T=0.8, seed `999 + update`).
  Save `best.pt` on a new best. Roll back if the score falls more than 2 combined standard errors below the best.
- **End jump:** at the end, try `W_start + k * (W_best - W_start)` for k in {1, 1.5, 2, 3, 5}. Score each on the
  held-out problems with 8 answers each and one shared seed. Keep the best candidate whose chat change is at most
  0.05. In both Kaggle races it kept k = 1, so no jump.

**Main settings** (`train.py` defaults for `--method granderl`):

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

## 6. GPU

Set `ENGINE_DEVICE=cuda`. Everything then runs in float32 (a T4 has no fast bf16); pass `--autocast 0`. Pin one run
per GPU with `CUDA_VISIBLE_DEVICES`. This path is what produced the Kaggle results above.

On a T4, a GrandeRL update takes about 4–9 s, and writing a chunk of about 150–185 answers takes about 30–36 s.

## 7. Pitfalls already fixed (do not reintroduce)

1. **Step direction.** `.grad` holds the gradient of the negative objective, so the step subtracts it. An earlier
   version added it.
2. **Full-precision checks.** The objective and KL checks must use float32 forward passes. In bf16, changes this small
   are below the rounding step, and the checks measure noise.
3. **Stochastic rounding.** On CPU, the bf16 copy that writes answers is refreshed with stochastic rounding, so small
   updates reach it on average.
4. **The sampler prior.** Fit it to the screening counts. A flat prior wastes about 75% of draws.
5. **Memory.** Peak is about 8 GB on CPU. Keep `--micro 4` on a 15 GB machine.
6. **Noisy choices.** Choosing checkpoints and jump sizes on the same problems with the same seed inflated one result
   by about 6 points. Hence the separate held-out set, the fresh seeds and the untouched test sets.
7. **Too few test answers.** HumanEval with 2 answers per problem showed a fake 1.5-point gap. Re-scored with 10, it
   vanished. Use at least 5–10 answers per problem when comparing.
8. **Big jumps.** Jumping multiplies all accumulated change, including noise, so the end jump is capped at chat change
   0.05.

## 8. Another model

1. Set `ENGINE_MODEL`.
2. Re-run the screening: `python3 -I bank.py screen --out data` (about 18 minutes on CPU for SmolLM2).
3. Run the self-check.
4. If the model's answers do not use a ` ```python ` block, adapt `engine.CLOSED` and `extract_code` in `evaluate.py`.
