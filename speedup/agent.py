"""agent.py - a ~1K-parameter neural network that alone decides how to make SmolLM-135M faster, and learns from it.

Every step the network sees the ENTIRE architecture of Smol and scores every possible action:
    for each layer:  remove it | prune 25% of its MLP neurons | prune 10% of its MLP neurons | drop one KV head group
    global:          heal (30 KL-distillation steps toward the original) | undo the last edit
It picks one (softmax), the action is applied to the real model, and REINFORCE updates the network.
No fixed schedule: when to cut, when to heal and when to undo are all the network's choices.

Reward (fixed, the network cannot change it):
    + 10 x measured speed change (real CPU timing, relative to the original)
    - 20 x the change in how far search perplexity is above the 10% budget
    - 0.3 per heal (healing costs time)
Safeguards (scoring side, not the agent): a model only becomes the saved best if it is the fastest so far AND
VALIDATION perplexity is within 10%; a model past 1.5x perplexity is rolled back.
"""
import json, os, time, statistics, torch, torch.nn as nn
import judge as J, edits as E, heal as H

OUT = f'{J.ROOT}/runs'; os.makedirs(OUT, exist_ok=True)
HOURS = float(os.environ.get('HOURS', 4))
TOL, BREAK = 1.10, 1.5
KINDS = ['layer', 'mlp25', 'mlp10', 'kv', 'heal', 'undo']
FEAT = len(KINDS) + 8

class Policy(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(FEAT, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
    def forward(self, f): return self.net(f).squeeze(-1)

def tok_s(m): return J.quick_speed(m)

def actions(m, mem, ppl_ratio, speed_ratio, can_undo):
    acts, feats = [], []; L = E.n_layers(m)
    glob = [ppl_ratio - 1, speed_ratio - 1, L / 30]
    for i in range(L):
        l = m.model.layers[i]; w = l.mlp.gate_proj.out_features / 1536; kv = E.n_kv(m, i)
        for kind, ok in [('layer', L > 1), ('mlp25', w > 0.3), ('mlp10', w > 0.15), ('kv', kv > 1)]:
            if not ok: continue
            c, g = mem.get((kind, i), (0.0, 0.0))                 # last measured ppl cost / speed gain of this action
            feats.append([k == kind for k in KINDS] + [i / max(L - 1, 1), w, kv / 3, c, g] + glob); acts.append((kind, i))
    for kind, ok in [('heal', True), ('undo', can_undo)]:
        if not ok: continue
        c, g = mem.get((kind, -1), (0.0, 0.0))
        feats.append([k == kind for k in KINDS] + [0, 0, 0, c, g] + glob); acts.append((kind, -1))
    return acts, torch.tensor(feats, dtype=torch.float32)

def main():
    torch.manual_seed(0); T0 = time.time()
    log = open(f'{OUT}/edit_log_v2.jsonl', 'a')
    def note(**k): k['t'] = round(time.time() - T0); log.write(json.dumps(k) + '\n'); log.flush(); print(k, flush=True)
    orig = J.load_original()
    P0, V0, S0 = J.perplexity(orig, 'search'), J.perplexity(orig, 'validation'), J.speed(orig)[0]
    Q0 = tok_s(orig)                                               # quick-timing baseline
    pol = Policy(); opt = torch.optim.Adam(pol.parameters(), lr=1e-2); base_r = 0.0
    note(event='start', search_ppl=P0, val_ppl=V0, tok_s=S0, policy_params=sum(p.numel() for p in pol.parameters()))
    m, p, s = orig, P0, Q0; history = []; mem = {}; heal_pos = 0; best = S0; step = 0
    while time.time() - T0 < HOURS * 3600:
        step += 1
        acts, f = actions(m, mem, p / P0, s / Q0, bool(history))
        dist = torch.distributions.Categorical(logits=pol(f)); a = dist.sample(); kind, i = acts[a.item()]
        if kind == 'heal':
            new, heal_pos = H.heal(m, steps=30, start=heal_pos, log=lambda x: None)
        elif kind == 'undo':
            new = history.pop()[0]
        elif kind == 'layer': new = E.remove_layer(m, i)
        elif kind in ('mlp25', 'mlp10'): new = E.prune_mlp(m, i, 0.25 if kind == 'mlp25' else 0.10, E.mlp_importance(m))
        else:
            new = min((E.prune_kv_group(m, i, g) for g in range(E.n_kv(m, i))), key=lambda c: J.perplexity(c, 'search'))
        p2 = J.perplexity(new, 'search')
        if kind != 'heal': s = tok_s(m); s2 = tok_s(new)               # paired: time current and edited back to back
        else: s2 = s
        over = lambda q: max(0.0, q / P0 - TOL)
        r = 10 * (s2 - s) / Q0 - 20 * (over(p2) - over(p)) - (0.3 if kind == 'heal' else 0.0)
        mem[(kind, i)] = (p2 / p - 1, (s2 - s) / Q0)
        loss = -(r - base_r) * dist.log_prob(a); base_r = 0.9 * base_r + 0.1 * r
        opt.zero_grad(); loss.backward(); opt.step()
        if p2 > BREAK * P0:
            note(step=step, action=[kind, i], search_ppl=round(p2, 2), reward=round(r, 3), event='rolled_back'); continue
        if kind not in ('heal', 'undo'): history.append((m, p, s))
        if kind == 'heal' and history: history[-1] = history[-1]       # heal keeps the undo point
        m, p, s = new, p2, s2
        note(step=step, action=[kind, i], search_ppl=round(p, 3), tok_s=round(s, 2), speedup=round(s / Q0, 3), layers=E.n_layers(m), reward=round(r, 3))
        if p <= TOL * P0 and s / Q0 > best / S0 * 1.02:                      # candidate best: confirm on validation + full timing
            v = J.perplexity(m, 'validation')
            if v <= TOL * V0:
                sf = J.speed(m)[0]
                if sf > best:
                    best = sf; torch.save(m, f'{OUT}/best_model_v2.pt')
                    note(event='new_best', val_ppl=round(v, 3), tok_s=round(sf, 2), speedup=round(sf / S0, 3), layers=E.n_layers(m), params=E.params(m))
        if step % 10 == 0: torch.save(pol.state_dict(), f'{OUT}/policy_v2.pt')
    note(event='done', best_tok_s=best, best_speedup=best / S0)

if __name__ == '__main__': main()
