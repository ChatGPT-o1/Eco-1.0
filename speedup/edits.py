"""edits.py - structural edits on a LlamaForCausalLM (SmolLM-135M). Each edit returns a new model; the input is untouched."""
import copy, torch, torch.nn as nn
import judge as J

def n_layers(m): return len(m.model.layers)

def _reindex(m):
    for i, l in enumerate(m.model.layers): l.self_attn.layer_idx = i
    m.config.num_hidden_layers = len(m.model.layers)
    if getattr(m.config, 'layer_types', None): m.config.layer_types = m.config.layer_types[:len(m.model.layers)]

def remove_layer(m, i):
    m = copy.deepcopy(m)
    m.model.layers = nn.ModuleList([l for j, l in enumerate(m.model.layers) if j != i]); _reindex(m)
    return m

@torch.no_grad()
def mlp_importance(m, batches=4):
    """Per layer, per neuron: mean |silu(gate)*up| x ||down column||, on healing text (not search/val/test)."""
    x = J.split('healing')[:batches * 8].to(J.DEVICE); acts = {}
    hooks = []
    for i, l in enumerate(m.model.layers):
        def h(mod, inp, i=i):
            a = inp[0].abs().mean(dim=(0, 1)); acts[i] = acts.get(i, 0) + a
        hooks.append(l.mlp.down_proj.register_forward_pre_hook(h))
    for b in range(0, len(x), 8): m(x[b:b + 8])
    for hk in hooks: hk.remove()
    return {i: acts[i] * l.mlp.down_proj.weight.norm(dim=0) for i, l in enumerate(m.model.layers)}

def _keep_rows(lin, idx, dim):
    w = lin.weight.data.index_select(dim, idx).clone()
    new = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype); new.weight.data = w; return new

@torch.no_grad()
def prune_mlp(m, i, frac, imp=None):
    """Remove the `frac` least-important MLP neurons of layer i. `imp` must come from THIS model (mlp_importance(m))."""
    m = copy.deepcopy(m); imp = (imp or mlp_importance(m))[i]
    mlp = m.model.layers[i].mlp; n = mlp.gate_proj.out_features
    keep = imp.to(mlp.gate_proj.weight.device).argsort(descending=True)[:max(1, int(round(n * (1 - frac))))].sort().values
    mlp.gate_proj = _keep_rows(mlp.gate_proj, keep, 0); mlp.up_proj = _keep_rows(mlp.up_proj, keep, 0)
    mlp.down_proj = _keep_rows(mlp.down_proj, keep, 1)
    return m

@torch.no_grad()
def prune_kv_group(m, i, g):
    """Remove KV group g of layer i (its 1 key/value head and the query heads that share it)."""
    m = copy.deepcopy(m); at = m.model.layers[i].self_attn
    hd = at.head_dim; rep = at.num_key_value_groups
    nkv = at.k_proj.out_features // hd
    if nkv <= 1: raise ValueError('cannot remove the last KV group')
    kv_keep = torch.tensor([h for h in range(nkv) if h != g])
    q_keep = torch.cat([torch.arange(h * rep * hd, (h + 1) * rep * hd) for h in kv_keep.tolist()])
    kvd = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in kv_keep.tolist()])
    q_keep, kvd = q_keep.to(at.q_proj.weight.device), kvd.to(at.q_proj.weight.device)
    at.q_proj = _keep_rows(at.q_proj, q_keep, 0); at.o_proj = _keep_rows(at.o_proj, q_keep, 1)
    at.k_proj = _keep_rows(at.k_proj, kvd, 0); at.v_proj = _keep_rows(at.v_proj, kvd, 0)
    return m

def n_kv(m, i):
    at = m.model.layers[i].self_attn; return at.k_proj.out_features // at.head_dim

def params(m): return sum(p.numel() for p in m.parameters())
