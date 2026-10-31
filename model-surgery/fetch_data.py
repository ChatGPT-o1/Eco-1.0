#!/usr/bin/env python3
"""Step 0 (optional): download a ready-made "code" domain dataset, split into counting and held-out parts.

Writes to OUT_DIR:
  count_code.jsonl     GitHub files {"text"} for token counting (repos split by hash, so no repo overlaps held-out)
  count_chat.jsonl     coding Q&A {"instruction","response"} for counting (all but the last 2000 rows)
  heldout_code.jsonl   up to --heldout-files GitHub files from held-out repos, first 1500 chars, {"text"}
  heldout_chat.jsonl   100 coding Q&A rows from the held-out tail
  heldout_general.jsonl 100 Wikipedia passages {"text"} (to see what non-code ability is lost)
  humaneval.parquet    164 HumanEval problems
Sources: codeparrot/github-code-clean (1 shard, ~360 MB), ise-uiuc/Magicoder-Evol-Instruct-110K (~255 MB),
Salesforce/wikitext (wikitext-103 test), openai/openai_humaneval. Check their licenses for your use.

  python -I fetch_data.py --out data/
"""
import argparse, hashlib, json, os, random, re, shutil

LANGS = ["Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "GO", "Rust", "Shell", "PHP",
         "Ruby", "SQL", "HTML", "CSS", "Markdown"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-files", type=int, default=300)
    ap.add_argument("--max-count-files", type=int, default=0, help="cap counting files (0 = whole shard)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    cache = os.path.join(a.out, "_downloads")
    get = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset", local_dir=cache)

    shutil.copy(get("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"),
                os.path.join(a.out, "humaneval.parquet"))

    rng = random.Random(0)
    pf = pq.ParquetFile(get("codeparrot/github-code-clean", "data/train-00000-of-00880.parquet"))
    held_repo = lambda r: int(hashlib.md5(r.encode()).hexdigest(), 16) % 10 == 0
    by_lang = {}
    n_count = 0
    with open(os.path.join(a.out, "count_code.jsonl"), "w") as f:
        for rb in pf.iter_batches(batch_size=2000, columns=["code", "repo_name", "language"]):
            for row in rb.to_pylist():
                if held_repo(row["repo_name"]):
                    if len(by_lang.setdefault(row["language"], [])) < 200:
                        by_lang[row["language"]].append(row["code"])
                elif not a.max_count_files or n_count < a.max_count_files:
                    f.write(json.dumps({"text": row["code"][:20000]}) + "\n")
                    n_count += 1
    per = max(1, a.heldout_files // len(LANGS))
    held = []
    for lang in LANGS:
        pool = [c for c in by_lang.get(lang, []) if len(c) >= 300]
        rng.shuffle(pool)
        held += [{"text": c[:1500], "lang": lang} for c in pool[:per]]
    with open(os.path.join(a.out, "heldout_code.jsonl"), "w") as f:
        for r in held:
            f.write(json.dumps(r) + "\n")

    rows = [json.loads(l) for l in open(get("ise-uiuc/Magicoder-Evol-Instruct-110K", "data-evol_instruct-decontaminated.jsonl"))]
    with open(os.path.join(a.out, "count_chat.jsonl"), "w") as f:
        for r in rows[:-2000]:
            f.write(json.dumps(r) + "\n")
    tail = [r for r in rows[-2000:] if 100 <= len(r["response"]) and len(r["instruction"]) + len(r["response"]) <= 4000]
    rng.shuffle(tail)
    with open(os.path.join(a.out, "heldout_chat.jsonl"), "w") as f:
        for r in tail[:100]:
            f.write(json.dumps(r) + "\n")

    wt = pq.read_table(get("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet")).column("text").to_pylist()
    paras = [p.strip() for p in wt if len(p.strip()) > 400 and not p.strip().startswith("=")]
    fix = lambda s: re.sub(r" @(.)@ ", r"\1", s).replace(" , ", ", ").replace(" . ", ". ").replace(" 's", "'s")
    step = max(1, len(paras) // 100)
    with open(os.path.join(a.out, "heldout_general.jsonl"), "w") as f:
        for p in paras[::step][:100]:
            f.write(json.dumps({"text": fix(p)[:1500]}) + "\n")
    shutil.rmtree(cache, ignore_errors=True)
    print(f"wrote {a.out}: {n_count} counting files, {len(held)} held-out files, {len(rows) - 2000} counting chats")


if __name__ == "__main__":
    main()
