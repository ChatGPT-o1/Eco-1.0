#!/usr/bin/env python3
"""Build a self-contained Kaggle script kernel that runs the llm2ncm pipeline on a free GPU.

Packs llm2ncm/ and model-surgery/ (sources only) into runner_template.py as a base64 tar.gz, and writes
kernels/<variant>/{run_llm2ncm.py,kernel-metadata.json}. The tarball is deterministic, so rebuilding
unchanged sources gives a byte-identical file.

  python build_kernel.py                    # builds both variants
  kaggle kernels push -p kernels/smoke --accelerator NvidiaTeslaT4   # ~10 min check of the GPU path
  kaggle kernels push -p kernels/full  --accelerator NvidiaTeslaT4   # the real run

See README.md for the full workflow (status polling, downloading output, asking the model questions).
"""
import argparse, base64, gzip, io, json, os, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC_DIRS = ("llm2ncm", "model-surgery", "draft")
KEEP = (".py", ".md", ".txt")

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
COMMON = {
    "base_model": BASE_MODEL,
    # versions that ran on Kaggle's image (torch 2.11+cu128, 2x T4) in an earlier project; requirements.txt only says >=5.0
    "pip_extra": ["transformers==5.19.0", "peft==0.21.2", "accelerate==1.15.0"],
    "chat_prompts": ["What is 2 + 2?", "What is the capital of France?", "Write a Python function that adds two numbers."],
    "chat_refine": 4,
    "chat_new": 128,
    "selftest": True,
    "mode": "pipeline",
}
VARIANTS = {
    "full": dict(COMMON, name="llm2ncm-smollm2-135m", pipeline_args=[
        "--device", "cuda", "--n", "50000", "--comp-steps", "5000", "--gen-steps", "5000", "--humaneval", "20"]),
    "smoke": dict(COMMON, name="llm2ncm-smollm2-135m-smoke", pipeline_args=[
        "--device", "cuda", "--n", "400", "--comp-steps", "30", "--gen-steps", "30", "--humaneval", "2", "--eval-prompts", "2"]),
}


def job(name, n, steps, note="", comp=(), gen=(), ev=(), recipe="base"):
    """One sweep job: train compressor + generator for `steps` each on n windows, then ncm_quick_eval."""
    return {"name": name, "recipe": recipe, "n": n, "note": note or f"{steps} steps",
            "comp": ["--steps", str(steps), *comp], "gen": ["--steps", str(steps), "--sample-every", "0", *gen], "eval": list(ev)}


def sweep(name, recipes, jobs):
    return dict(COMMON, name=name, mode="sweep", recipes=recipes, jobs=jobs)


# Sweeps A/B training changes at a small budget on both T4s (see README.md). Longest jobs first so the two GPUs stay busy.
VARIANTS["sweep-baseline"] = sweep("llm2ncm-sweep-baseline", {"base": {"n": 32000}}, [
    job("base_n8000_s6000", 8000, 6000),
    job("base_n2000_s6000", 2000, 6000),
    job("base_n32000_s2000", 32000, 2000),
    job("base_n8000_s2000", 8000, 2000),
    job("base_n2000_s2000", 2000, 2000),
])
# tiny CPU-sized sweep for LLM2NCM_LOCAL=1 dry runs of the runner itself
VARIANTS["sweep-local-test"] = sweep("llm2ncm-sweep-local-test", {"base": {"n": 96}}, [
    job("t_a", 64, 4, "mask-draws 2, lr-final", comp=["--lr-final", "0.1"], gen=["--mask-draws", "2"], ev=["--n", "6"]),
    job("t_b", 32, 4, "second job", comp=["--split-seed", "0", "--seed", "1"], gen=["--batch", "4", "--seed", "1"], ev=["--n", "6"]),
])


def payload():
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tf:
        for d in SRC_DIRS:
            for root, dirs, files in os.walk(os.path.join(REPO, d)):
                dirs[:] = sorted(x for x in dirs if x != "__pycache__")
                for f in sorted(files):
                    if f.endswith(KEEP):
                        full = os.path.join(root, f)
                        ti = tf.gettarinfo(full, arcname=os.path.relpath(full, REPO))
                        ti.uid = ti.gid = 0
                        ti.uname = ti.gname = ""
                        ti.mtime = 0
                        with open(full, "rb") as fh:
                            tf.addfile(ti, fh)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "\n".join(b64[i:i + 100] for i in range(0, len(b64), 100))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", default="timcook123", help="Kaggle username that owns the kernel")
    a = ap.parse_args()
    template = open(os.path.join(HERE, "runner_template.py")).read()
    data = payload()
    for variant, cfg in VARIANTS.items():
        out = os.path.join(HERE, "kernels", variant)
        os.makedirs(out, exist_ok=True)
        code = template.replace("@@CONFIG@@", json.dumps(cfg, indent=1)).replace("@@PAYLOAD@@", "\n" + data + "\n")
        open(os.path.join(out, "run_llm2ncm.py"), "w").write(code)
        meta = {
            "id": f"{a.owner}/{cfg['name']}", "title": cfg["name"], "code_file": "run_llm2ncm.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4",
        }
        json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
        print(f"{variant}: {meta['id']}  {os.path.getsize(os.path.join(out, 'run_llm2ncm.py')) / 1024:.0f} KB")




# ---- GPU smoke of the sweep runner: one tiny job per distinct flag set (catches arg/CUDA errors in ~5 min before a long sweep)
VARIANTS["sweep-smoke"] = sweep("llm2ncm-sweep-smoke", {
    "base": {"n": 300}, "distill1": {"n": 300, "args": ["--distill-frac", "1.0"]}, "enc18": {"n": 300, "args": ["--enc-layer", "18"]},
}, [
    job("s_ctl", 300, 20, "control seed1", comp=["--split-seed", "0", "--seed", "1"], gen=["--seed", "1"], ev=["--n", "40"]),
    job("s_mask_lr", 300, 20, "mask-draws 4 + lr-final", comp=["--split-seed", "0", "--lr-final", "0.1"], gen=["--mask-draws", "4"], ev=["--n", "40"]),
    job("s_slots", 300, 20, "scales 4,16 cb4096", comp=["--split-seed", "0", "--scales", "4,16", "--codebook", "4096"], ev=["--n", "40"]),
    job("s_dist", 300, 20, "distill 1.0", recipe="distill1", comp=["--split-seed", "0"], ev=["--n", "40"]),
    job("s_enc", 300, 20, "enc-layer 18", recipe="enc18", comp=["--split-seed", "0"], ev=["--n", "40"]),
])


# ---- improvement sweep: every job is the baseline plus ONE change at n=8000 windows, 2000 steps each, on the SAME
# held-out windows (--split-seed 0), so differences are due to the change (see ctl_seed* for run-to-run noise)
def imp(name, note, recipe="base", comp=(), gen=()):
    return job(name, 8000, 2000, note, recipe=recipe, comp=["--split-seed", "0", *comp], gen=list(gen), ev=["--n", "400"])


VARIANTS["sweep-improve"] = sweep("llm2ncm-sweep-improve", {
    "base": {"n": 8000}, "distill1": {"n": 8000, "args": ["--distill-frac", "1.0"]}, "enc18": {"n": 8000, "args": ["--enc-layer", "18"]},
}, [
    imp("mask4", "gen --mask-draws 4", gen=["--mask-draws", "4"]),
    imp("slots4_16", "comp --scales 4,16", comp=["--scales", "4,16"]),
    imp("mask2", "gen --mask-draws 2", gen=["--mask-draws", "2"]),
    imp("ctl_seed0", "control (baseline code)"),
    imp("ctl_seed1", "control, seed 1 (noise)", comp=["--seed", "1"], gen=["--seed", "1"]),
    imp("lrdecay", "comp --lr-final 0.1", comp=["--lr-final", "0.1"]),
    imp("corrupt30", "gen --corrupt 0.3", gen=["--corrupt", "0.3"]),
    imp("genlr1e3", "gen --lr 1e-3", gen=["--lr", "1e-3"]),
    imp("cb4096", "comp --codebook 4096", comp=["--codebook", "4096"]),
    imp("distill100", "build --distill-frac 1.0", recipe="distill1"),
    imp("enc18", "build --enc-layer 18", recipe="enc18"),
])


# ---- v2: on-policy answer windows (the original model's own answers to chat prompts, cut where the NCM writes them)
# + 100% distilled code/chat windows + compressor LR decay. Two variants train side by side, one per T4, on the same data.
V1_REFERENCE = ("NCM refine=4 (judge bits/token / agreement / distinct-2 / HumanEval): v1 (50k windows, original recipe) 4.768 / 0.060 / 0.23 / 0/20; "
                "v2 (VQ, 2+8 slots) 5.237 / 0.084 / 0.384 / 0/20; v2-wide (VQ, 4+16 slots) 4.920 / 0.083 / 0.439 / 0/20; "
                "the original model 0.546 / 1.0 / 0.688 / 2/20. Compressor token recovery from codes (1 level / 2 levels): v1 9.6% / 20.4%, "
                "v2 11.3% / 22.5%, v2-wide 13.5% / 43.1%.")


def variants(name, data_args, common_args, vs):
    return dict(COMMON, name=name, mode="variants", data_args=data_args, common_args=common_args, variants=vs, reference=V1_REFERENCE)


V2_DATA = ["--n", "24000", "--prompt-frac", "0.5", "--distill-frac", "1.0", "--answer-blocks", "3"]
V2_COMMON = ["--device", "cuda", "--comp-steps", "5000", "--gen-steps", "5000", "--humaneval", "20", "--comp-extra=--lr-final 0.1",
             "--gen-extra=--slot-emb"]
VARIANTS["v2"] = variants("llm2ncm-v2", V2_DATA, V2_COMMON, [
    {"name": "v2", "note": "v2 data + per-slot embeddings + compressor LR decay (codes 2+8)", "args": []},
    {"name": "v2-wide", "note": "same, with twice the code slots (4+16): 2 tokens per fine code instead of 4", "args": ["--scales", "4,16"]},
])
# ---- v3: the four design changes from the v2 diagnosis, all on top of the v2 recipe (same data, plus the teacher's top-8 distribution):
# FSQ codes (no dead codebook entries), a predictability term in compressor training, an autoregressive first code level, and soft
# distillation of the teacher's distribution. v3 keeps the 2+8 code slots (apples to apples with v2); v3-wide uses 4+16.
V3_DATA = ["--n", "24000", "--prompt-frac", "0.5", "--distill-frac", "1.0", "--answer-blocks", "3", "--soft-k", "8"]
V3_ARGS = ["--comp-extra=--quant fsq --fsq-levels 8,5,5,5 --pred-weight 0.1 --lr-final 0.1",
           "--gen-extra=--slot-emb --ar-first-level --soft-weight 0.5"]
VARIANTS["v3"] = variants("llm2ncm-v3", V3_DATA, ["--device", "cuda", "--comp-steps", "5000", "--gen-steps", "5000", "--humaneval", "20"], [
    {"name": "v3", "note": "FSQ + predictability loss + autoregressive outline + soft distillation (codes 2+8)", "args": V3_ARGS},
    {"name": "v3-wide", "note": "same, with twice the code slots (4+16)", "args": ["--scales", "4,16", *V3_ARGS]},
])
VARIANTS["v2-local-test"] = variants("llm2ncm-v2-local-test", ["--n", "48", "--prompt-frac", "0.5", "--distill-frac", "1.0", "--answer-blocks", "2"],
                                     ["--device", "cpu", "--comp-steps", "4", "--gen-steps", "4", "--humaneval", "0", "--eval-prompts", "2",
                                      "--sample-every", "2", "--comp-extra=--lr-final 0.1", "--gen-extra=--slot-emb"], [
    {"name": "a", "note": "local test a", "args": []},
    {"name": "b", "note": "local test b", "args": ["--scales", "4,16"]},
])


# ---- final: the long run. Two models side by side, one per T4, each building its own data:
#   final-b32  32-word blocks, 4+16 code slots (the best compressor so far: 43% of words recoverable from codes)
#   final-b16  16-word blocks, 4+8 slots: half as much to predict per block, aimed at the failure every run showed (the codes)
# Recipe = what measured best: the original model's own answers as data, per-slot embeddings, soft distillation, compressor LR
# decay; plus 4x the batch, fp16 (self-checked against fp32), 80k windows, and a clock-driven learning-rate schedule so the
# generator trains until the deadline (7 h after launch minus the evaluation time) instead of for a guessed number of steps.
V3_REFERENCE = ("; v3 (FSQ, 2+8) 5.331 / 0.097 / 0.351 / 0/20; v3-wide (FSQ, 4+16) 4.425 / 0.078 / 0.366 / 0/20")


def final_cfg(name, n, comp_steps, humaneval, eval_prompts, sample_every, batch=32, device="cuda", gen_time="", save_every=2000, **extra):
    common = ["--device", device, "--comp-steps", str(comp_steps), "--gen-steps", "1000000", "--batch", str(batch),
              "--humaneval", str(humaneval), "--eval-prompts", str(eval_prompts), "--sample-every", str(sample_every),
              "--refine", "1,2,4", "--comp-extra=--lr-final 0.1",
              f"--gen-extra=--slot-emb --soft-weight 0.5 --amp --save-every {save_every} --log-every 50{gen_time}"]

    def data(block, answer_blocks):
        return ["--n", str(n), "--prompt-frac", "0.5", "--distill-frac", "1.0", "--answer-blocks", str(answer_blocks),
                "--soft-k", "8", "--block", str(block)]

    vs = [{"name": "final-b32", "note": "32-word blocks, 4+16 code slots", "data_args": data(32, 3), "args": ["--scales", "4,16"]},
          {"name": "final-b16", "note": "16-word blocks, 4+8 code slots", "data_args": data(16, 6), "args": ["--scales", "4,8"]}]
    return dict(COMMON, name=name, mode="variants", data_args=[], common_args=common, variants=vs,
                reference=V1_REFERENCE + V3_REFERENCE, **extra)


VARIANTS["final"] = final_cfg("llm2ncm-final", 100000, 20000, 40, 64, 500, budget_hours=7.0, eval_reserve_min=28, ckpt_every_min=30)
VARIANTS["final-smoke"] = final_cfg("llm2ncm-final-smoke", 1200, 300, 2, 8, 100, gen_time=" --time-budget-min 4", save_every=100, ckpt_every_min=1)
VARIANTS["final-local-test"] = final_cfg("llm2ncm-final-local-test", 48, 4, 0, 2, 3, batch=4, device="cpu", gen_time=" --time-budget-min 0.4",
                                         save_every=3, ckpt_every_min=0.1)


# ---- draft head (EAGLE-style): exact output, faster decoding. Two heads side by side, one per T4, same data:
#   eagle-ttt4  trained with a 4-step rollout ("training-time test", EAGLE-3) so drafts 2..k are trained for real use
#   eagle-ttt1  plain one-step training (EAGLE-1), the control
def draft_cfg(name, n, max_new, minutes, bench_n, bench_max_new, ks, device="cuda", batch=16, bench_dtypes=("float32", "float16")):
    return dict(COMMON, name=name, mode="draft", selftest_script="draft/selftest_draft.py",
                data_args=["--n", str(n), "--max-new", str(max_new), "--batch", "128" if device == "cuda" else "16"],
                train_args=["--time-budget-min", str(minutes), "--batch", str(batch), "--eval-every", "500"]
                           + (["--amp"] if device == "cuda" else []),
                bench_args=["--n", str(bench_n), "--max-new", str(bench_max_new), "--k", ks],
                bench_dtypes=list(bench_dtypes),
                heads=[{"name": "eagle-ttt4", "note": "4-step training-time test (EAGLE-3 style)", "args": ["--steps-ahead", "4"]},
                       {"name": "eagle-ttt1", "note": "one-step training (EAGLE-1 style), control", "args": ["--steps-ahead", "1"]}])


VARIANTS["draft"] = draft_cfg("llm2ncm-draft", 12000, 256, 50, 40, 192, "3,5")
VARIANTS["draft-local-test"] = draft_cfg("llm2ncm-draft-local-test", 24, 32, 0.5, 2, 24, "2", device="cpu", batch=4, bench_dtypes=("float32",))


if __name__ == "__main__":
    main()
