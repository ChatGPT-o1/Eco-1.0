"""agent.py - a ~1K-parameter neural network that edits SmolLM-135M to make it faster, and learns from the results.

The policy sees ALL of Smol: for every possible edit (remove layer i / prune 25% of layer i's MLP / drop a KV group of
layer i) it gets a feature vector describing that layer and the edit, scores it with the same small MLP, and samples
an edit from a softmax over all of them. REINFORCE updates the policy from the reward of each edit.

Reward  = predicted speed gain of the edit (latency model fit from real timings)
          - penalty when search perplexity goes over budget.
Healing = every few accepted edits, KL-distill toward the original (heal.py).
Gate    = after healing, VALIDATION perplexity must be within 10% of the original and real timing is measured;
          only then does the model become a checkpoint. A failed gate reverts to the last checkpoint and is penalized.
The agent can change Smol's weights and architecture. It cannot touch judge.py, the data, or this reward.
"""
import json, os, time, copy, torch, torch.nn as nn
import judge as J, edits as E, heal as H

OUT = f'{J.ROOT}/runs'; os.makedirs(OUT, exist_ok=True)
HOURS = float(os.environ.get('HOURS', 3))
TOL, HARD = 1.10, 1.25          # final budget; per-edit hard limit before healing
HEAL_EVERY, HEAL_STEPS = 3, 60

# ---- latency model (ms/token), fit from floor.py: 0 layers 2.9, full 30.3, all-MLP-50% 24.1
FIXED, LAYER_OVH, MLP_MS, KV_MS = 2.9, 0.36, 0.413, 0.046
def latency(m):
    ms = FIXED
    for l in m.model.layers:
        ms += LAYER_OVH + MLP_MS * l.mlp.gate_proj.out_features / 1536 + KV_MS * 3 * (l.self_attn.k_proj.out_features // 64) / 3
    return ms

# ---- the 1K-parameter policy
FEAT = 11
class Policy(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(FEAT, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
    def forward(self, f): return self.net(f).squeeze(-1)

scan = json.load(open(os.path.join(os.path.dirname(__file__), 'scan_results.json')))

def actions(m, cost, ppl_ratio, since_heal):
    """All valid edits on the current model, with features."""
    acts, feats = [], []; L = E.n_layers(m)
    for i in range(L):
        l = m.model.layers[i]; w = l.mlp.gate_proj.out_features / 1536; kv = E.n_kv(m, i)
        cand = [('layer', i, LAYER_OVH + MLP_MS * w + KV_MS * kv)]
        if w > 0.3: cand.append(('mlp', i, MLP_MS * 0.25 * w))
        if kv > 1: cand.append(('kv', i, KV_MS))
        for kind, _, gain in cand:
            onehot = [kind == 'layer', kind == 'mlp', kind == 'kv']
            feats.append(onehot + [i / max(L - 1, 1), w, kv / 3, cost.get((kind, i), 0.0), gain, ppl_ratio - 1, since_heal / HEAL_EVERY, L / 30])
            acts.append((kind, i, gain))
    return acts, torch.tensor(feats, dtype=torch.float32)

def apply(m, kind, i):
    if kind == 'layer': return E.remove_layer(m, i)
    if kind == 'mlp': return E.prune_mlp(m, i, 0.25, E.mlp_importance(m))
    best, bp = None, 1e9
    for g in range(E.n_kv(m, i)):            # drop the cheapest group (measured on search)
        c = E.prune_kv_group(m, i, g); p = J.perplexity(c, 'search')
        if p < bp: best, bp = c, p
    return best

def main():
    torch.manual_seed(0)
    log = open(f'{OUT}/edit_log.jsonl', 'a')
    def note(**k): k['t'] = round(time.time() - T0); log.write(json.dumps(k) + '\n'); log.flush(); print(k, flush=True)
    T0 = time.time()
    orig = J.load_original()
    P0, V0, MS0 = J.perplexity(orig, 'search'), J.perplexity(orig, 'validation'), latency(orig)
    S0 = J.speed(orig)[0]
    note(event='start', search_ppl=P0, val_ppl=V0, tok_s=S0, policy_params=sum(p.numel() for p in Policy().parameters()))
    pol = Policy(); opt = torch.optim.Adam(pol.parameters(), lr=1e-2); baseline = 0.0
    # cost memory seeded from the single-edit scan (relative ppl increase); updated by every measured edit
    cost = {}
    for i in range(30):
        cost[('layer', i)] = min(scan['layer'][str(i)] / P0 - 1, 3.0)
        cost[('mlp', i)] = (scan['mlp50'][str(i)] / P0 - 1) / 2
        cost[('kv', i)] = scan['kv'][str(i)] / P0 - 1
    m, ckpt = orig, orig; since, heal_pos, pending = 0, 0, []
    while time.time() - T0 < HOURS * 3600:
        p_now = J.perplexity(m, 'search')
        acts, f = actions(m, cost, p_now / P0, since)
        logits = pol(f); dist = torch.distributions.Categorical(logits=logits); a = dist.sample()
        kind, i, gain = acts[a.item()]
        cand = apply(m, kind, i); p = J.perplexity(cand, 'search')
        cost[(kind, i)] = p / p_now - 1                       # what this edit actually cost
        if p > HARD * P0:
            r = -5.0; note(event='reject', edit=[kind, i], search_ppl=p)
        else:
            r = 10 * gain / MS0 - 20 * max(0.0, p / P0 - TOL)
            m = cand; since += 1; pending.append((f, a))
            note(event='accept', edit=[kind, i], search_ppl=round(p, 3), layers=E.n_layers(m), pred_speedup=round(MS0 / latency(m), 3))
        loss = -(r - baseline) * dist.log_prob(a); baseline = 0.9 * baseline + 0.1 * r
        opt.zero_grad(); loss.backward(); opt.step()
        if since >= HEAL_EVERY or (since and p > 1.15 * P0):
            m, heal_pos = H.heal(m, steps=HEAL_STEPS, start=heal_pos, log=lambda s: None)
            v = J.perplexity(m, 'validation')
            if v > TOL * V0:                                   # one more healing round, then judge
                m, heal_pos = H.heal(m, steps=HEAL_STEPS, start=heal_pos, log=lambda s: None); v = J.perplexity(m, 'validation')
            if v <= TOL * V0:
                tps = J.speed(m)[0]; ckpt = m; torch.save(m, f'{OUT}/best_model.pt')
                note(event='gate_pass', val_ppl=round(v, 3), tok_s=round(tps, 2), real_speedup=round(tps / S0, 3), layers=E.n_layers(m), params=E.params(m))
                gr = 2.0
            else:
                note(event='gate_fail', val_ppl=round(v, 3)); m = ckpt; gr = -5.0
            if pending:                                        # credit the edits of this round with the gate result
                l2 = -gr * sum(torch.distributions.Categorical(logits=pol(ff)).log_prob(aa) for ff, aa in pending) / len(pending); opt.zero_grad(); l2.backward(); opt.step()
            pending, since = [], 0
            torch.save(pol.state_dict(), f'{OUT}/policy.pt')
    note(event='done')

if __name__ == '__main__': main()
