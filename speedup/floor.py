"""floor.py - what can pruning buy at all? Time the full model, 0 layers, 15 layers, and all MLPs cut 50%."""
import torch, torch.nn as nn, judge as J, edits as E
def keep_layers(m, idx):
    import copy; m = copy.deepcopy(m); m.model.layers = nn.ModuleList([m.model.layers[i] for i in idx]); E._reindex(m); return m
m = J.load_original(); imp = E.mlp_importance(m)
mlp50 = m
for i in range(30): mlp50 = E.prune_mlp(mlp50, i, 0.5, imp)
cases = {'full (30 layers)': m, '15 layers': keep_layers(m, range(0, 30, 2)), '0 layers': keep_layers(m, []),
         '30 layers, MLP -50%': mlp50}
for name, x in cases.items():
    r, se = J.speed(x)
    print(f'{name:22s} {r:6.1f} tok/s  ({1000/r:5.1f} ms/token)  params {E.params(x)/1e6:.1f}M', flush=True)
