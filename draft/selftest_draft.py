#!/usr/bin/env python3
"""Checks for the draft head and the speculative decoder (CPU, a few minutes):
  1. exactness: with an UNTRAINED head, speculative decoding returns exactly plain greedy's tokens (k = 1, 3, 6)
  2. alignment: after the head memorises one answer, decoding that prompt accepts most drafted tokens
     (catches off-by-one errors between training targets and how drafts are used) and is still exact

  python -I selftest_draft.py MODEL_DIR
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from eagle import EagleHead, draft_loss, greedy_plain, greedy_spec


def main():
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    lm = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    eos = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>",) if tok.convert_tokens_to_ids(t) is not None}
    qs = ["What is 2 + 2?", "What is the capital of France?", "Write a Python function that adds two numbers."]
    prompts = [tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True),
                   add_special_tokens=False)["input_ids"] for q in qs]

    head = EagleHead(lm.config)
    head.init_from(lm)
    head.eval()
    for kk in (1, 3, 6):
        for p in prompts:
            ref, _ = greedy_plain(lm, p, 48, eos)
            got, passes, acc = greedy_spec(lm, head, p, 48, eos, k=kk)
            assert got == ref, f"k={kk}: speculative output differs from plain greedy\n{ref}\n{got}"
    print("1. untrained head: speculative output == plain greedy for k = 1, 3, 6 on 3 prompts")

    # the training-time-test mask: step 1 causal; step r sees true keys up to j-r+1 plus one key per later step
    from eagle import ttt_mask
    m = ttt_mask(5, 3, "cpu").int()
    assert m[4].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], m[4].tolist()
    assert ttt_mask(4, 1, "cpu").equal(torch.ones(4, 4, dtype=torch.bool).tril())

    p0 = prompts[2]
    ref, plain_passes = greedy_plain(lm, p0, 64, eos)
    seq = torch.tensor([p0 + ref])
    valid = torch.ones_like(seq, dtype=torch.bool)
    for steps in (1, 4):
        head = EagleHead(lm.config)
        head.init_from(lm)
        head.train()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        for _ in range(200):
            loss, st = draft_loss(lm, head, seq, valid, torch.tensor([len(p0)]), steps=steps)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        got1, _, acc1 = greedy_spec(lm, head, p0, 64, eos, k=1)
        got4, passes4, acc4 = greedy_spec(lm, head, p0, 64, eos, k=4)
        assert got1 == ref and got4 == ref, "memorised head: output differs from plain greedy"
        assert all(a == 1 for a in acc1), f"k=1 on the memorised answer must accept every draft (alignment), got {acc1}"
        mean4 = sum(acc4) / len(acc4)
        print(f"{2 + (steps > 1)}. memorised one answer with {steps} training step(s): first draft always accepted; with 4 drafts "
              f"{mean4:.2f} accepted per round, {len(got4) / passes4:.2f} tokens per pass (plain {len(ref) / plain_passes:.2f}); exact")
        if steps > 1:
            assert mean4 > 3.0, "multi-step training should make drafts 2..4 match on a memorised answer: check ttt_mask"
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
