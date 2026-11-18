import json, judge as J, edits as E
m = J.load_original(); imp = E.mlp_importance(m); base = J.perplexity(m, 'search'); out = {'base': base, 'layer': {}, 'mlp50': {}, 'kv': {}}
for i in range(E.n_layers(m)):
    out['layer'][i] = J.perplexity(E.remove_layer(m, i), 'search')
    out['mlp50'][i] = J.perplexity(E.prune_mlp(m, i, 0.5, imp), 'search')
    out['kv'][i] = min(J.perplexity(E.prune_kv_group(m, i, g), 'search') for g in range(3))
    print(i, round(out['layer'][i], 2), round(out['mlp50'][i], 2), round(out['kv'][i], 2), flush=True)
    json.dump(out, open('scan.json', 'w'))
