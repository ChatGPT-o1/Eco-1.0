#!/usr/bin/env python3
"""Step 2: count how often each token id appears in a domain corpus.

Sources (repeat any flag; all counts are summed):
  --parquet PATH:COLUMN            one document per row (e.g. GitHub code files)
  --jsonl PATH:FIELD[,FIELD...]    fields joined with a newline
  --chat-jsonl PATH:USER:ASSISTANT rendered through the model's chat template, so
                                   template/special tokens are counted too
  --text GLOB                      one document per file
Keep held-out data OUT of these sources; use --skip-last N to hold out the tail of jsonl files.

  python -I count_tokens.py --model DIR --out counts.npy --parquet code.parquet:code \
      --chat-jsonl magicoder.jsonl:instruction:response --skip-last 2000
"""
import argparse, glob, json, time

import numpy as np


def batches(it, n=1000):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def read_jsonl(path, skip_last):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if skip_last:
        lines = lines[:-skip_last]
    for line in lines:
        if line.strip():
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model dir whose tokenizer.json is used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parquet", action="append", default=[])
    ap.add_argument("--jsonl", action="append", default=[])
    ap.add_argument("--chat-jsonl", action="append", default=[])
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--skip-last", type=int, default=0, help="hold out the last N rows of each jsonl source")
    ap.add_argument("--max-chars", type=int, default=20000, help="truncate each document")
    ap.add_argument("--max-docs", type=int, default=0, help="cap documents per source (0 = all)")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{a.model}/tokenizer.json")
    V = tok.get_vocab_size(with_added_tokens=True)
    V = max(V, max(tok.get_vocab(with_added_tokens=True).values()) + 1)
    counts = np.zeros(V, dtype=np.int64)
    stats = {}
    t0 = time.time()

    def consume(name, docs):
        nd = nt = 0
        for b in batches(docs):
            b = [d[: a.max_chars] for d in b]
            for e in tok.encode_batch(b, add_special_tokens=False):
                ids = np.asarray(e.ids, dtype=np.int64)
                counts[: V] += np.bincount(ids, minlength=V)[:V]
                nt += len(ids)
            nd += len(b)
            if a.max_docs and nd >= a.max_docs:
                break
        stats[name] = {"docs": nd, "tokens": nt}
        print(f"{name}: {nd} docs, {nt} tokens ({time.time() - t0:.0f}s)", flush=True)

    for spec in a.parquet:
        path, col = spec.rsplit(":", 1)
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        def docs():
            for rb in pf.iter_batches(batch_size=1000, columns=[col]):
                for t in rb.column(0).to_pylist():
                    yield t or ""
        consume(spec, docs())

    for spec in a.jsonl:
        path, fields = spec.rsplit(":", 1)
        fs = fields.split(",")
        consume(spec, ("\n".join(str(r.get(f, "")) for f in fs) for r in read_jsonl(path, a.skip_last)))

    if a.chat_jsonl:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(a.model)
        for spec in a.chat_jsonl:
            path, uf, af = spec.rsplit(":", 2)
            def docs():
                for r in read_jsonl(path, a.skip_last):
                    msgs = [{"role": "user", "content": r[uf]}, {"role": "assistant", "content": r[af]}]
                    yield hf.apply_chat_template(msgs, tokenize=False)
            consume(spec, docs())

    for g in a.text:
        def docs():
            for p in sorted(glob.glob(g, recursive=True)):
                try:
                    yield open(p, encoding="utf-8", errors="replace").read()
                except (IsADirectoryError, PermissionError):
                    continue
        consume(g, docs())

    if not stats:
        ap.error("give at least one source")
    np.save(a.out, counts)
    json.dump(stats, open(a.out + ".json", "w"), indent=2)
    print(f"saved {a.out}: {int((counts > 0).sum())} of {V} ids seen, {int(counts.sum())} tokens")


if __name__ == "__main__":
    main()
