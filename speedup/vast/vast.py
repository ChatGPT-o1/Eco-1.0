"""vast.py - minimal Vast.ai client over HTTPS (the API key is injected by the session's network proxy, never stored here).
    python vast/vast.py search ['Tesla T4'] [num_gpus]      cheapest offers
    python vast/vast.py create <offer_id> <onstart.sh> [disk_gb] [image]
    python vast/vast.py show                                 my instances
    python vast/vast.py logs <instance_id> [tail]
    python vast/vast.py destroy <instance_id>
"""
import json, os, sys, time, requests
API = 'https://console.vast.ai/api/v0'; CA = os.environ.get('REQUESTS_CA_BUNDLE') or ('/root/.ccr/ca-bundle.crt' if os.path.exists('/root/.ccr/ca-bundle.crt') else True)
def call(method, path, body=None, params=None):
    r = requests.request(method, API + path, json=body, params=params, verify=CA, timeout=60)
    if r.status_code >= 400: raise SystemExit(f'{method} {path} -> {r.status_code}: {r.text[:500]}')
    return r.json() if r.text.strip() else {}
def search(gpu='Tesla T4', n=2, limit=15):
    q = {'verified': {'eq': True}, 'external': {'eq': False}, 'rentable': {'eq': True}, 'rented': {'eq': False}, 'gpu_name': {'eq': gpu},
         'num_gpus': {'eq': int(n)}, 'order': [['dph_total', 'asc']], 'type': 'on-demand', 'limit': limit, 'allocated_storage': 20.0}
    return call('POST', '/bundles/', q)['offers']
def create(offer, onstart, disk=20, image='pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime', label='smol-speedup'):
    body = {'client_id': 'me', 'image': image, 'env': {}, 'price': None, 'disk': float(disk), 'label': label, 'extra': None, 'onstart': onstart,
            'image_login': None, 'python_utf8': False, 'lang_utf8': False, 'use_jupyter_lab': False, 'jupyter_dir': None, 'force': False,
            'cancel_unavail': False, 'template_hash_id': None, 'user': None, 'runtype': 'ssh'}
    return call('PUT', f'/asks/{offer}/', body)
def show():
    r = requests.get('https://console.vast.ai/api/v1/instances/', verify=CA, timeout=60); r.raise_for_status(); return r.json().get('instances', [])
def logs(iid, tail=200):
    rj = call('PUT', f'/instances/request_logs/{iid}/', {'tail': str(tail)}); url = rj.get('result_url')
    if not url: return json.dumps(rj)
    for _ in range(40):
        time.sleep(0.5); r = requests.get(url, verify=CA, timeout=30)
        if r.status_code == 200: return r.text
    return ''
def destroy(iid): return call('DELETE', f'/instances/{iid}/', {})
if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'search':
        for o in search(*(sys.argv[2:4] or ['Tesla T4'])):
            print({k: (round(o[k], 4) if isinstance(o.get(k), float) else o.get(k)) for k in ('id', 'gpu_name', 'num_gpus', 'dph_total', 'reliability2', 'geolocation',
                   'cuda_max_good', 'cpu_cores_effective', 'cpu_ram', 'disk_space', 'inet_down', 'duration', 'driver_version', 'storage_cost', 'inet_down_cost')})
    elif c == 'create': print(create(sys.argv[2], open(sys.argv[3]).read(), *(sys.argv[4:6])))
    elif c == 'show':
        for i in show(): print({k: i.get(k) for k in ('id', 'actual_status', 'cur_state', 'intended_status', 'gpu_name', 'num_gpus', 'dph_total', 'label', 'status_msg', 'start_date', 'ssh_host', 'ssh_port')})
    elif c == 'logs': print(logs(sys.argv[2], *(sys.argv[3:4])))
    elif c == 'destroy': print(destroy(sys.argv[2]))
