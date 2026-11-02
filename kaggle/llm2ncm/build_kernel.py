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
SRC_DIRS = ("llm2ncm", "model-surgery")
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
}
VARIANTS = {
    "full": dict(COMMON, name="llm2ncm-smollm2-135m", pipeline_args=[
        "--device", "cuda", "--n", "50000", "--comp-steps", "5000", "--gen-steps", "5000", "--humaneval", "20"]),
    "smoke": dict(COMMON, name="llm2ncm-smollm2-135m-smoke", pipeline_args=[
        "--device", "cuda", "--n", "400", "--comp-steps", "30", "--gen-steps", "30", "--humaneval", "2", "--eval-prompts", "2"]),
}


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


if __name__ == "__main__":
    main()
