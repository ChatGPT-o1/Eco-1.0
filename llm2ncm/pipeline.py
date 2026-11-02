#!/usr/bin/env python3
"""One command: convert a Hugging Face causal LM into an experimental Next-Concept Model and report how
much quality and speed it kept versus the original. Each stage's output under --work is reused, so a
crashed run resumes.

  data.pt  ->  compressor.pt  ->  generator/  ->  eval.json  ->  REPORT.md

  python -I ../model-surgery/fetch_data.py --out msdata/          # once: code + coding-chat data
  python -I pipeline.py --model HuggingFaceTB/SmolLM2-135M-Instruct --data msdata/ --work runs/smol-ncm
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    cmd = [sys.executable, "-I", os.path.join(HERE, script)] + [str(x) for x in args]
    print("\n$", " ".join(cmd), flush=True)
    t = time.time()
    r = subprocess.run(cmd)
    print(f"[{script} exit {r.returncode}, {time.time() - t:.0f}s]", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=6000, help="training windows")
    ap.add_argument("--prefix", type=int, default=64)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--scales", default="2,8")
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--distill-frac", type=float, default=0.5)
    ap.add_argument("--comp-steps", type=int, default=1500)
    ap.add_argument("--gen-steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--backbone", default="auto", choices=["auto", "llm", "small"])
    ap.add_argument("--eval-prompts", type=int, default=16)
    ap.add_argument("--refine", default="1,2,4")
    ap.add_argument("--humaneval", type=int, default=10)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    a = ap.parse_args()
    W = a.work
    os.makedirs(W, exist_ok=True)
    data, comp, gdir, ev = f"{W}/data.pt", f"{W}/compressor.pt", f"{W}/generator", f"{W}/eval.json"

    if not os.path.exists(data):
        run("build_data.py", "--model", a.model, "--data", a.data, "--out", data, "--n", a.n, "--prefix", a.prefix,
            "--block", a.block, "--distill-frac", a.distill_frac, "--threads", a.threads, "--device", a.device)
    if not os.path.exists(comp + ".json"):
        run("train_compressor.py", "--model", a.model, "--data", data, "--out", comp, "--scales", a.scales,
            "--codebook", a.codebook, "--steps", a.comp_steps, "--threads", a.threads, "--device", a.device)
    gcfg = f"{gdir}/config.json"
    if not os.path.exists(gcfg) or json.load(open(gcfg)).get("trained_steps") != a.gen_steps:
        run("train_generator.py", "--model", a.model, "--data", data, "--compressor", comp, "--out", gdir,
            "--backbone", a.backbone, "--lora-r", a.lora_r, "--steps", a.gen_steps, "--batch", a.batch, "--threads", a.threads, "--device", a.device)
    if not os.path.exists(ev):
        run("ncm_eval.py", "--model", a.model, "--generator", gdir, "--data", a.data, "--out", ev,
            "--n-prompts", a.eval_prompts, "--refine", a.refine, "--humaneval", a.humaneval, "--threads", a.threads, "--device", a.device)

    c = json.load(open(comp + ".json"))
    g = json.load(open(gcfg))
    e = json.load(open(ev))
    last = c["history"][-1] if c["history"] else {}
    lines = [
        f"# LLM → NCM conversion report: {a.model}", "",
        f"Block = {c['cfg']['L']} tokens, code levels {c['cfg']['scales']} (codebook {c['cfg']['codebook']}), "
        f"prefix {a.prefix}, {a.n} training windows ({a.distill_frac:.0%} distilled from the original model).",
        f"Generator backbone: **{g['backbone']}** ({g['backbone_note']}), {g['trained_steps']} steps, batch {g['batch']}.",
        f"Passes per block = {len(c['cfg']['scales'])} code levels + R token passes (+1 to cache the finished block); "
        f"the original model needs {c['cfg']['L']} passes per block.", "",
        "## Quality and speed vs the original model", "", e["table"], "",
        "Judge bits/token: the original model's surprise at each output, lower = closer to what it would write. "
        "The original's own greedy output is the floor; real held-out text shows what normal human-written text scores.", "",
        "## Compressor", "",
        f"Token reconstruction accuracy from codes alone (held-out): "
        + ", ".join(f"{k.split('_')[1]} level(s): {v:.1%}" for k, v in last.items() if k.endswith("token_acc"))
        + f"; codebook use per level: {last.get('codebook_use')}.", "",
        "## Sample outputs (first prompt)", "",
    ]
    for name, samples in e["samples"].items():
        lines += [f"**{name}**", "", "```", (samples[0] if samples else "")[:600], "```", ""]
    report = "\n".join(lines)
    open(f"{W}/REPORT.md", "w").write(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
