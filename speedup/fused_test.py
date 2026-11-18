"""fused_test.py - on a GPU: is the fused integer-weight decoder exact, how fast is it, where does its time go,
and what does each storage format cost in perplexity?"""
import json, math, time, traceback, torch
import cupy as cp
import judge as J, smol as S, fastgen as G, quant as Q, fused as F

def say(**k): print(json.dumps(k), flush=True)
dev = 'cuda:0'
from transformers import AutoModelForCausalLM
P = S.extract(AutoModelForCausalLM.from_pretrained(J.MODEL_DIR, dtype=torch.float32))
tok = J.tokenizer(); prompt = tok(J.PROMPTS[0])['input_ids']
ident = lambda net: net.set_rank([torch.arange(net.dff)] * net.NL)
net0 = S.Net(P, dev); ident(net0); ids = J.split('test').to(dev); T = ids.shape[1]
with torch.no_grad(): base = net0.seq_nll(ids, net0.full())
ref = G.Decoder(S.materialize(net0, net0.full())).capture(); ref_tps = G.tok_per_s(ref, prompt)
h = net0.full(); h.half = True; ref16 = G.Decoder(S.materialize(net0, h)).capture(); ref16_tps = G.tok_per_s(ref16, prompt)
say(event='reference', gpu=torch.cuda.get_device_name(0), cupy=cp.__version__, torch_decoder_fp32_tok_s=round(ref_tps, 1), torch_decoder_fp16_tok_s=round(ref16_tps, 1),
    test_ppl=round(math.exp(base.sum().item() / (base.numel() * (T - 1))), 3), text=tok.decode(ref.generate(prompt, 40)))
del ref, ref16

def check(name, fmt_of, rpb=4, breakdown=False):
    """Quantize, compare the fused decoder with PyTorch running the same rounded weights, time it, measure perplexity."""
    out = dict(event='fused', model=name, rows_per_block=rpb if not callable(rpb) else 'tuned', MB_per_token=round(Q.bytes_per_token(P, fmt_of) / 1e6, 1))
    try:
        arrs, Pq = Q.quantize_model(P, fmt_of, dev); netq = S.Net(Pq, dev); ident(netq)
        with torch.no_grad():
            d = (netq.seq_nll(ids, netq.full()) - base) / (T - 1)
            x = netq.embed(torch.tensor([prompt], device=dev))
            for i in range(netq.NL): x = netq.layer(i, x, netq.full())
            ref_lg = netq.logits(x)[0, -1].float().cpu()
        tdec = G.Decoder(S.materialize(netq, netq.full())).capture(); ttoks = tdec.generate(prompt, 64); del tdec, netq
        out.update(test_ppl_change_pct=round(100 * (math.exp(d.mean().item()) - 1), 3), se_pct=round(100 * d.std().item() / math.sqrt(len(d)), 3))
        dec = F.Fused(P, arrs, rpb=rpb)
        lg = dec.first_logits(prompt); toks = dec.generate(prompt, 64); tps = dec.tok_per_s(prompt)
        out.update(logits_max_rel_err=float((lg - ref_lg).abs().max() / ref_lg.abs().max()), tokens_same=f'{sum(a == b for a, b in zip(toks, ttoks))}/64',
                   tok_s=round(tps, 1), ms_per_token=round(1000 / tps, 3), speedup_vs_torch_fp32=round(tps / ref_tps, 2), launches=len(dec.launches),
                   text=tok.decode(toks[:40]))
        if breakdown: out['ms_by_kernel'] = dec.breakdown()
        del dec
    except Exception:
        out['error'] = traceback.format_exc()[-2500:]
    cp.get_default_memory_pool().free_all_blocks(); torch.cuda.empty_cache()
    say(**out)

check('everything 16-bit', lambda n: 'i16', breakdown=True)
check('everything 8-bit', lambda n: 'i8', breakdown=True)
check('everything 4-bit', lambda n: 'n4', breakdown=True)
check('8-bit, attention 4-bit', lambda n: 'n4' if n != 'E' and n[1] in ('q', 'k', 'v', 'o') else 'i8')
check('8-bit, MLP 4-bit', lambda n: 'n4' if n != 'E' and n[1] in ('g', 'u', 'dn') else 'i8')
for r in (1, 2, 8):
    check('everything 8-bit', lambda n: 'i8', rpb=r)
say(event='done')
