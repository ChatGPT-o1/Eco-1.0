#!/usr/bin/env python3
"""Builds a Kaggle kernel that races plain RL (GRPO) against GrandeRL-1.0 on 2 T4 GPUs, then runs the full test.

  python3 build_race.py STAGE OUTDIR KAGGLE_USERNAME
  kaggle kernels push -p OUTDIR --accelerator NvidiaTeslaT4 -t 5400

STAGE: smoke (4 min each, 12-problem test) | ten (10 min each, full test, 2 answers per problem) |
       seed1 (as ten, other training seed, HumanEval 10 answers per problem) | full (120 min each, 5 answers)
Needs a Kaggle account with GPU quota; the kernel downloads SmolLM2 and the datasets itself (internet on)."""
import base64, json, os, sys
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the GrandeRL-1.0/ folder
FILES = {
    "src/crl/smol_code.py": f"{PKG}/crl/smol_code.py",
    "src/crl/evaluate.py": f"{PKG}/crl/evaluate.py",
    "src/crl/v8/engine.py": f"{PKG}/crl/v8/engine.py",
    "src/crl/v8/train.py": f"{PKG}/crl/v8/train.py",
    "src/crl/v8/evalx.py": f"{PKG}/crl/v8/evalx.py",
    "src/crl/v8/compare.py": f"{PKG}/crl/v8/compare.py",
    "src/crl/v8/data/screen.json": f"{PKG}/crl/v8/data/screen.json",
    "scr/local/msdata/heldout_prompts.jsonl": f"{PKG}/scr/local/msdata/heldout_prompts.jsonl",
}
stage, out, user = sys.argv[1], sys.argv[2], sys.argv[3]
CFG = dict(smoke=dict(budget=4, dev_every=3, dev_n=1, eval_limit=12, eval_n=1),
           ten=dict(budget=10, dev_every=5, dev_n=4, eval_limit=0, eval_n=2),
           seed1=dict(budget=10, dev_every=5, dev_n=4, eval_limit=0, eval_n=2, he_n=10, train_seed=1, eval_seed=2025),
           full=dict(budget=120, dev_every=10, dev_n=4, eval_limit=0, eval_n=5))[stage]
blob = {k: base64.b64encode(open(v, "rb").read()).decode() for k, v in FILES.items()}
runner = r'''
import base64, json, os, subprocess, sys, time
CFG = %s
FILES = %s
W = "/kaggle/working"
T0 = time.time()
def log(*a):
    print(f"[race {time.time() - T0:7.0f}s]", *a, flush=True)
for k, v in FILES.items():
    p = os.path.join(W, k); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(base64.b64decode(v))
log("pip:", subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.19.0"], capture_output=True, text=True).returncode)
env = dict(os.environ, ENGINE_DEVICE="cuda", ENGINE_SCR=W + "/scr", ENGINE_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct",
           HF_HOME=W + "/hf", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
PY = [sys.executable, "-I"]
V8 = W + "/src/crl/v8"
os.makedirs(W + "/runs", exist_ok=True); os.makedirs(W + "/evals", exist_ok=True)
def start(args, gpu, logname):
    e = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
    return subprocess.Popen(PY + args, stdout=open(W + "/" + logname, "w"), stderr=subprocess.STDOUT, env=e, cwd=V8)
common = ["--budget-min", str(CFG["budget"]), "--threads", "2", "--micro", "8", "--autocast", "0", "--pick-split", "1",
          "--no-teacher", "1", "--dev-every", str(CFG["dev_every"]), "--dev-n", str(CFG["dev_n"]),
          "--seed", str(CFG.get("train_seed", 0))]
log("training: GRPO on GPU 0, GrandeRL-1.0 on GPU 1, budget", CFG["budget"], "min each")
pg = start([V8 + "/train.py", "--method", "grpo", "--out", W + "/runs/grpo"] + common, 0, "train_grpo.log")
pv = start([V8 + "/train.py", "--method", "granderl", "--out", W + "/runs/v14"] + common, 1, "train_v14.log")
for name, p in (("grpo", pg), ("v14", pv)):
    log(name, "training exit", p.wait())
def ev(model, out, gpu):
    args = [V8 + "/evalx.py", "--model", model, "--out", W + "/evals/" + out + ".json", "--threads", "2", "--n", str(CFG["eval_n"]),
            "--seed", str(CFG.get("eval_seed", 2024)), "--he-n", str(CFG.get("he_n", 0))]
    if CFG["eval_limit"]:
        args += ["--limit", str(CFG["eval_limit"])]
    return start(args, gpu, "eval_" + out + ".log")
log("evaluation")
jobs = []
if os.path.exists(W + "/runs/grpo/final/config.json"):
    jobs.append(ev(W + "/runs/grpo/final", "grpo", 0))
if os.path.exists(W + "/runs/v14/final/config.json"):
    jobs.append(ev(W + "/runs/v14/final", "v14", 1))
# V14 before its end jump (best checkpoint only): the end jump spends extra compute after the shared budget
exp = """
import sys, torch; sys.path.insert(0, '%%s')
import engine as E
tok, m = E.load(); m.load_state_dict(torch.load('%%s', map_location='cuda'))
m.save_pretrained('%%s'); tok.save_pretrained('%%s')
""" %% (V8, W + "/runs/v14/best.pt", W + "/runs/v14/nojump", W + "/runs/v14/nojump")
if os.path.exists(W + "/runs/v14/best.pt"):
    log("export v14 without end jump:", subprocess.run(PY + ["-c", exp], env=dict(env, CUDA_VISIBLE_DEVICES="1"), cwd=V8).returncode)
for j in jobs:
    j.wait()
jobs = [ev("HuggingFaceTB/SmolLM2-135M-Instruct", "base", 0)]
if os.path.exists(W + "/runs/v14/nojump/config.json"):
    jobs.append(ev(W + "/runs/v14/nojump", "v14_nojump", 1))
for j in jobs:
    j.wait()
files = [W + "/evals/" + n + ".json" for n in ("base", "grpo", "v14_nojump", "v14") if os.path.exists(W + "/evals/" + n + ".json")]
r = subprocess.run(PY + [V8 + "/compare.py"] + files, capture_output=True, text=True, env=env, cwd=V8)
open(W + "/compare.txt", "w").write(r.stdout + r.stderr)
log("COMPARE\n" + r.stdout + r.stderr)
for n in ("train_grpo.log", "train_v14.log", "eval_grpo.log", "eval_v14.log", "eval_v14_nojump.log", "eval_base.log"):
    if os.path.exists(W + "/" + n):
        lines = [l for l in open(W + "/" + n, errors="replace") if any(x in l for x in ("round", "update", "DEV", "finished", "Error", "Traceback", "mbpp_test", "humaneval", "end jump"))]
        log("---", n, "(%%d matching lines)" %% len(lines)); print("".join(lines[-25:]), flush=True)
for d in ("runs/grpo", "runs/v14"):
    for f in ("ckpt_a.pt", "ckpt_b.pt", "best.pt", "best.pt.tmp"):
        p = os.path.join(W, d, f)
        if os.path.exists(p):
            os.remove(p)
log("done")
''' % (json.dumps(CFG), json.dumps(blob))
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "race.py"), "w").write(runner)
json.dump({"id": f"{user}/granderl-vs-grpo-{stage}", "title": f"granderl-vs-grpo-{stage}", "code_file": "race.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
print("built", stage, "->", out, f"({os.path.getsize(os.path.join(out, 'race.py')) // 1024} KB)")
