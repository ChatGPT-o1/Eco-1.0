"""test_exact.py - checks that the fast paths agree with HuggingFace and with each other (CPU, fp32).
1. smol.Net perplexity of the original model == HuggingFace perplexity.
2. A masked architecture (smol.Net) == the same edits done by physical surgery on the HF model (edits.py).
3. The decoder running materialized (physically smaller) weights picks exactly the tokens the masked forward predicts.
4. Evaluating an edit from the cached prefix == evaluating it from scratch.
"""
import copy, math, random, torch, torch.nn as nn
import judge as J, smol as S, fastgen as G, edits as E

torch.manual_seed(0); random.seed(0)
hf = J.load_original(); P = S.extract(hf); net = S.Net(P, 'cpu')
ids = J.split('search')[:6]; T = ids.shape[1]
with torch.no_grad():
    # 1
    lg = hf(ids).logits[:, :-1].float()
    ref = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1), reduction='sum').item()
    mine = net.seq_nll(ids, net.full()).sum().item()
    print(f'1. original: HF nll {ref:.3f}  smol.Net nll {mine:.3f}  rel diff {abs(ref - mine) / ref:.2e}')
    assert abs(ref - mine) / ref < 1e-5
    # usage statistics -> which neurons are kept
    st = net.stats(J.split('calib')[:8]); rank = [v.argsort(descending=True) for v in st['imp']]; net.set_rank(rank)
    # 2: remove layer 7 entirely, MLP of layer 3 to 70 %, drop KV group 1 of layer 12
    arch = net.full().apply(('attn', 7)).apply(('mlp', 7)).apply(('p30', 3)).apply(('kv', 12, 1))
    m = copy.deepcopy(hf)
    mlp = m.model.layers[3].mlp; keep = rank[3][:round(1536 * 0.7)].sort().values
    mlp.gate_proj = E._keep_rows(mlp.gate_proj, keep, 0); mlp.up_proj = E._keep_rows(mlp.up_proj, keep, 0); mlp.down_proj = E._keep_rows(mlp.down_proj, keep, 1)
    m = E.prune_kv_group(m, 12, 1); m = E.remove_layer(m, 7)
    lg = m(ids).logits[:, :-1].float()
    ref = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1), reduction='sum').item()
    mine = net.seq_nll(ids, arch).sum().item()
    print(f'2. edited:   HF-surgery nll {ref:.3f}  masked nll {mine:.3f}  rel diff {abs(ref - mine) / ref:.2e}')
    assert abs(ref - mine) / ref < 1e-5
    # 3: decoder on materialized weights vs masked forward, for a heavily edited architecture
    arch = net.full()
    for e in [('attn', 2), ('attn', 20), ('mlp', 25), ('p10', 0), ('p30', 5), ('p30', 5), ('kv', 9, 0), ('kv', 9, 2), ('kv', 28, 1), ('attn', 29), ('mlp', 29)]:
        arch = arch.apply(e)
    dec = G.Decoder(S.materialize(net, arch))
    prompt = J.tokenizer()(J.PROMPTS[0])['input_ids']; gen = dec.generate(prompt, 24)
    seq = torch.tensor([prompt + gen]); x = net.embed(seq)
    for i in range(net.NL): x = net.layer(i, x, arch)
    pred = net.logits(x)[0].argmax(-1).tolist()
    ok = all(pred[len(prompt) - 1 + j] == gen[j] for j in range(24))
    print('3. decoder tokens == masked-forward predictions:', ok, '|', repr(J.tokenizer().decode(gen)[:60])); assert ok
    full_M = S.materialize(net, net.full()); print('   params: full', S.n_params(full_M), ' edited', S.n_params(S.materialize(net, arch)))
    # 4: prefix cache
    base = net.seq_nll(ids, net.full())
    acts = [net.embed(ids)]
    for i in range(net.NL): acts.append(net.layer(i, acts[-1], net.full()))
    cur = net.full(); worst = 0.0
    for e in [('p10', 22), ('attn', 14), ('kv', 3, 2), ('mlp', 27), ('p30', 22), ('kv', 3, 0)]:
        new = cur.apply(e); i = e[1]; x = acts[i]; tmp = {}
        for j in range(i, net.NL): x = net.layer(j, x, new); tmp[j + 1] = x
        a = net.nll(x, ids); b = net.seq_nll(ids, new); worst = max(worst, ((a - b).abs() / b).max().item())
        for j, v in tmp.items(): acts[j] = v
        cur = new
    print(f'4. cached-prefix vs from-scratch: worst rel diff {worst:.2e}'); assert worst < 1e-5
print('ALL EXACT')
