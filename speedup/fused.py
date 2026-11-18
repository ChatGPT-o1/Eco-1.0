"""fused.py - SmolLM's token step as custom GPU kernels that read integer weights, replayed as one CUDA graph.

Why: with PyTorch ops a token costs ~45 tiny GPU launches per layer and the weights are read as 16/32-bit floats.
Here every weight matrix is read once per token by one kernel, straight from 16-, 8- or 4-bit integers (quant.py),
so the time per token follows the number of weight bytes. The small steps (norm, rotary position, attention over
the cache, activation, argmax) are one kernel each.

    dec = Fused(P, arrs)                 # P: smol.extract() weights, arrs: quant.quantize_model(...)[0]
    dec.generate(prompt_ids, 64)         # greedy tokens
    dec.tok_per_s(prompt_ids)            # timing
"""
import time
import numpy as np, cupy as cp, torch

SRC = r'''
#define LD4I(p) (*reinterpret_cast<const int4*>(p))
#define LD4F(p) (*reinterpret_cast<const float4*>(p))

// ---------------------------------------------------------------- weight matrix x vector, from integer weights
// y[r] (+)= scale[r] * sum_j W[r][j] * x[j].  One warp per row; each lane takes every 32nd 16-byte chunk of the row,
// so a warp reads 512 contiguous bytes at a time.
extern "C" __global__ void gemv_i16(const short* __restrict__ W, const float* __restrict__ scale, const float* __restrict__ x,
                                    float* __restrict__ y, const int rows, const int n, const int add)
{
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const short* row = W + (size_t)r * n;
    float acc = 0.f;
    for (int c = lane * 8; c < n; c += 256) {
        const int4 w = LD4I(row + c); const float4 a = LD4F(x + c), b = LD4F(x + c + 4);
        acc += (float)(short)(w.x) * a.x + (float)(w.x >> 16) * a.y + (float)(short)(w.y) * a.z + (float)(w.y >> 16) * a.w
             + (float)(short)(w.z) * b.x + (float)(w.z >> 16) * b.y + (float)(short)(w.w) * b.z + (float)(w.w >> 16) * b.w;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) { const float v = acc * scale[r]; y[r] = add ? y[r] + v : v; }
}

__device__ __forceinline__ float dot4b(const int w, const float4 x) {
    return (float)(signed char)(w) * x.x + (float)(signed char)(w >> 8) * x.y + (float)(signed char)(w >> 16) * x.z + (float)(w >> 24) * x.w;
}
extern "C" __global__ void gemv_i8(const signed char* __restrict__ W, const float* __restrict__ scale, const float* __restrict__ x,
                                   float* __restrict__ y, const int rows, const int n, const int add)
{
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const signed char* row = W + (size_t)r * n;
    float acc = 0.f;
    for (int c = lane * 16; c < n; c += 512) {
        const int4 w = LD4I(row + c);
        acc += dot4b(w.x, LD4F(x + c)) + dot4b(w.y, LD4F(x + c + 4)) + dot4b(w.z, LD4F(x + c + 8)) + dot4b(w.w, LD4F(x + c + 12));
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) { const float v = acc * scale[r]; y[r] = add ? y[r] + v : v; }
}

__device__ __forceinline__ float dot8n(const unsigned int w, const float* xp) {
    const float4 a = LD4F(xp), b = LD4F(xp + 4);
    return (float)((int)(w & 15u) - 8) * a.x + (float)((int)((w >> 4) & 15u) - 8) * a.y
         + (float)((int)((w >> 8) & 15u) - 8) * a.z + (float)((int)((w >> 12) & 15u) - 8) * a.w
         + (float)((int)((w >> 16) & 15u) - 8) * b.x + (float)((int)((w >> 20) & 15u) - 8) * b.y
         + (float)((int)((w >> 24) & 15u) - 8) * b.z + (float)((int)(w >> 28) - 8) * b.w;
}
// 4-bit weights (nibble = q + 8), 32 weights per 16-byte chunk, one signed 16-bit scale code per chunk
extern "C" __global__ void gemv_n4(const unsigned char* __restrict__ W, const short* __restrict__ bs, const float* __restrict__ scale,
                                   const float* __restrict__ x, float* __restrict__ y, const int rows, const int n, const int add)
{
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int nch = n >> 5;
    const unsigned char* row = W + (size_t)r * (n >> 1);
    const short* bsr = bs + (size_t)r * nch;
    float acc = 0.f;
    for (int ch = lane; ch < nch; ch += 32) {
        const int4 w = LD4I(row + ch * 16); const float* xp = x + ch * 32;
        acc += (float)bsr[ch] * (dot8n((unsigned int)w.x, xp) + dot8n((unsigned int)w.y, xp + 8) + dot8n((unsigned int)w.z, xp + 16) + dot8n((unsigned int)w.w, xp + 24));
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) { const float v = acc * scale[r]; y[r] = add ? y[r] + v : v; }
}

// ---------------------------------------------------------------- embedding row of the current token -> x   (one block of n threads)
extern "C" __global__ void embed_i16(const short* __restrict__ E, const float* __restrict__ scale, const int* __restrict__ tok, float* __restrict__ x, const int n)
{ const int t = tok[0], j = threadIdx.x; x[j] = (float)E[(size_t)t * n + j] * scale[t]; }
extern "C" __global__ void embed_i8(const signed char* __restrict__ E, const float* __restrict__ scale, const int* __restrict__ tok, float* __restrict__ x, const int n)
{ const int t = tok[0], j = threadIdx.x; x[j] = (float)E[(size_t)t * n + j] * scale[t]; }
extern "C" __global__ void embed_n4(const unsigned char* __restrict__ E, const short* __restrict__ bs, const float* __restrict__ scale,
                                    const int* __restrict__ tok, float* __restrict__ x, const int n)
{
    const int t = tok[0], j = threadIdx.x; const unsigned char b = E[(size_t)t * (n >> 1) + (j >> 1)];
    const int nib = (j & 1) ? (b >> 4) : (b & 15);
    x[j] = (float)(nib - 8) * ((float)bs[(size_t)t * (n >> 5) + (j >> 5)] * scale[t]);
}

// ---------------------------------------------------------------- RMS norm: h = x * rsqrt(mean(x^2) + eps) * w   (one block of n threads)
extern "C" __global__ void prep(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ h, const int n, const float eps)
{
    __shared__ float red[32];
    const int t = threadIdx.x; const float v = x[t]; float s = v * v;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((t & 31) == 0) red[t >> 5] = s;
    __syncthreads();
    float tot = 0.f; const int nw = blockDim.x >> 5;
    for (int i = 0; i < nw; i++) tot += red[i];
    h[t] = v * rsqrtf(tot / (float)n + eps) * w[t];
}

// ---------------------------------------------------------------- attention
// rotate this token's keys and store keys and values in the cache.   grid = key/value groups, block = 64
extern "C" __global__ void kvwrite(const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ cs, const float* __restrict__ sn,
                                   float* __restrict__ KC, float* __restrict__ VC, const int* __restrict__ posp, const int L)
{
    const int g = blockIdx.x, i = threadIdx.x, pos = posp[0];
    const float kv = k[g * 64 + i]; const float other = (i < 32) ? -k[g * 64 + i + 32] : k[g * 64 + i - 32];
    const size_t o = ((size_t)g * L + pos) * 64 + i;
    KC[o] = kv * cs[pos * 64 + i] + other * sn[pos * 64 + i];
    VC[o] = v[g * 64 + i];
}
// one block per query head, one thread per cache position (block = L threads, L a multiple of 32, <= 1024)
extern "C" __global__ void attn(const float* __restrict__ q, const float* __restrict__ cs, const float* __restrict__ sn,
                                const float* __restrict__ KC, const float* __restrict__ VC, float* __restrict__ out,
                                const int* __restrict__ posp, const int L, const int rep)
{
    __shared__ float qr[64]; __shared__ float wgt[1024]; __shared__ float red[32];
    const int hh = blockIdx.x, g = hh / rep, t = threadIdx.x, pos = posp[0], nw = blockDim.x >> 5;
    if (t < 64) {
        const float qv = q[hh * 64 + t]; const float other = (t < 32) ? -q[hh * 64 + t + 32] : q[hh * 64 + t - 32];
        qr[t] = qv * cs[pos * 64 + t] + other * sn[pos * 64 + t];
    }
    __syncthreads();
    float sc = -1e30f;
    if (t <= pos) {
        const float* kp = KC + ((size_t)g * L + t) * 64; float d = 0.f;
        for (int i = 0; i < 64; i++) d += qr[i] * kp[i];
        sc = d * 0.125f;
    }
    float m = sc;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((t & 31) == 0) red[t >> 5] = m;
    __syncthreads();
    float mx = -1e30f;
    for (int i = 0; i < nw; i++) mx = fmaxf(mx, red[i]);
    __syncthreads();
    const float e = (t <= pos) ? expf(sc - mx) : 0.f;
    float s = e;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((t & 31) == 0) red[t >> 5] = s;
    wgt[t] = e;
    __syncthreads();
    float tot = 0.f;
    for (int i = 0; i < nw; i++) tot += red[i];
    if (t < 64) {
        const float* vp = VC + (size_t)g * L * 64 + t; float o = 0.f;
        for (int p = 0; p <= pos; p++) o += wgt[p] * vp[(size_t)p * 64];
        out[hh * 64 + t] = o / tot;
    }
}

// ---------------------------------------------------------------- MLP activation: a = silu(g) * u
extern "C" __global__ void act(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ a, const int n)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { const float v = g[i]; a[i] = v / (1.f + expf(-v)) * u[i]; } }

// ---------------------------------------------------------------- greedy pick: per-block winners, then the overall winner
extern "C" __global__ void argmax1(const float* __restrict__ lg, float* __restrict__ bm, int* __restrict__ bi, const int n)
{
    __shared__ float sm[32]; __shared__ int si[32];
    const int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    float v = (i < n) ? lg[i] : -3.4e38f; int idx = i;
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o); const int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
    }
    if ((t & 31) == 0) { sm[t >> 5] = v; si[t >> 5] = idx; }
    __syncthreads();
    if (t == 0) {
        float bv = sm[0]; int bx = si[0];
        for (int w = 1; w < (blockDim.x >> 5); w++) if (sm[w] > bv || (sm[w] == bv && si[w] < bx)) { bv = sm[w]; bx = si[w]; }
        bm[blockIdx.x] = bv; bi[blockIdx.x] = bx;
    }
}
// writes the next token, records it, advances the position   (one block of nb threads, nb a multiple of 32)
extern "C" __global__ void argmax2(const float* __restrict__ bm, const int* __restrict__ bi, int* __restrict__ tok, int* __restrict__ pos,
                                   int* __restrict__ outbuf, const int nb)
{
    __shared__ float sm[32]; __shared__ int si[32];
    const int t = threadIdx.x;
    float v = (t < nb) ? bm[t] : -3.4e38f; int idx = (t < nb) ? bi[t] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o); const int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
    }
    if ((t & 31) == 0) { sm[t >> 5] = v; si[t >> 5] = idx; }
    __syncthreads();
    if (t == 0) {
        float bv = sm[0]; int bx = si[0];
        for (int w = 1; w < (blockDim.x >> 5); w++) if (sm[w] > bv || (sm[w] == bv && si[w] < bx)) { bv = sm[w]; bx = si[w]; }
        const int p = pos[0]; tok[0] = bx; outbuf[p] = bx; pos[0] = p + 1;
    }
}
'''
_K = {}
def K(name):
    if name not in _K: _K[name] = cp.RawKernel(SRC, name)
    return _K[name]

i32 = np.int32


def _cp(t):
    """A CuPy-owned copy of a torch tensor."""
    return cp.asarray(t.contiguous()).copy() if t.is_cuda else cp.asarray(t.detach().contiguous().numpy())


class Fused:
    def __init__(self, P, arrs, max_len=128, rpb=4, device=0):
        self.dev = cp.cuda.Device(device); self.L = max_len; assert max_len % 32 == 0 and max_len <= 1024
        self.rpb = rpb if callable(rpb) else (lambda rows, n, fmt: rpb)
        with self.dev:
            d = P['E'].shape[1]; V = P['E'].shape[0]; hd, nq, nkv = P['hd'], P['nq'], P['nkv']; dff = P['layers'][0]['g'].shape[0]
            assert hd == 64 and d % 32 == 0 and d <= 1024
            self.d, self.V = d, V; f32 = cp.float32
            z = lambda n, dt=f32: cp.zeros(n, dt)
            self.x, self.h, self.q, self.k, self.v, self.oin = z(d), z(d), z(nq * hd), z(nkv * hd), z(nkv * hd), z(nq * hd)
            self.g, self.u, self.a, self.lg = z(dff), z(dff), z(dff), z(V)
            nb = -(-V // 256); self.nb = -(-nb // 32) * 32; self.bm, self.bi = z(self.nb), z(self.nb, cp.int32)
            self.tok, self.pos, self.out = z(1, cp.int32), z(1, cp.int32), z(max_len, cp.int32)
            inv = 1.0 / (P['theta'] ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
            fr = np.outer(np.arange(max_len, dtype=np.float32), inv); emb = np.concatenate([fr, fr], -1)
            cs, sn = cp.asarray(np.cos(emb).astype(np.float32)), cp.asarray(np.sin(emb).astype(np.float32))
            self.keep = [cs, sn]; eps = np.float32(P['eps']); self.launches = []
            A = {name: (fmt, {k: _cp(v) for k, v in a.items()}) for name, (fmt, a) in arrs.items()}; self.A = A

            def gemv(name, rows, n, x, y, add):
                fmt, a = A[name]; r = self.rpb(rows, n, fmt)
                args = (a['q'], a['code'], a['scale'], x, y, i32(rows), i32(n), i32(add)) if fmt == 'n4' else (a['q'], a['scale'], x, y, i32(rows), i32(n), i32(add))
                self.launches.append((K('gemv_' + fmt), ((rows + r - 1) // r,), (32 * r,), args))

            def norm(w):
                wv = _cp(w.float()); self.keep.append(wv)
                self.launches.append((K('prep'), (1,), (d,), (self.x, wv, self.h, i32(d), eps)))

            fmt, a = A['E']
            self.launches.append((K('embed_' + fmt), (1,), (d,), ((a['q'], a['code'], a['scale'], self.tok, self.x, i32(d)) if fmt == 'n4'
                                                                  else (a['q'], a['scale'], self.tok, self.x, i32(d)))))
            for i, l in enumerate(P['layers']):
                KC, VC = z(nkv * max_len * hd), z(nkv * max_len * hd); self.keep += [KC, VC]
                norm(l['n1'])
                gemv((i, 'q'), nq * hd, d, self.h, self.q, 0); gemv((i, 'k'), nkv * hd, d, self.h, self.k, 0); gemv((i, 'v'), nkv * hd, d, self.h, self.v, 0)
                self.launches.append((K('kvwrite'), (nkv,), (64,), (self.k, self.v, cs, sn, KC, VC, self.pos, i32(max_len))))
                self.launches.append((K('attn'), (nq,), (max_len,), (self.q, cs, sn, KC, VC, self.oin, self.pos, i32(max_len), i32(nq // nkv))))
                gemv((i, 'o'), d, nq * hd, self.oin, self.x, 1)
                norm(l['n2'])
                gemv((i, 'g'), dff, d, self.h, self.g, 0); gemv((i, 'u'), dff, d, self.h, self.u, 0)
                self.launches.append((K('act'), (-(-dff // 256),), (256,), (self.g, self.u, self.a, i32(dff))))
                gemv((i, 'dn'), d, dff, self.a, self.x, 1)
            norm(P['nf'])
            gemv('E', V, d, self.h, self.lg, 0)
            self.launches.append((K('argmax1'), (nb,), (256,), (self.lg, self.bm, self.bi, i32(V))))
            self.launches.append((K('argmax2'), (1,), (self.nb,), (self.bm, self.bi, self.tok, self.pos, self.out, i32(nb))))

            self.stream = cp.cuda.Stream(non_blocking=True)
            with self.stream:
                self._step(); self.stream.synchronize()                 # compiles every kernel
                self.stream.begin_capture(); self._step(); self.graph = self.stream.end_capture()
                self.pos[...] = 0; self.stream.synchronize()

    def _step(self):
        for k, grid, block, args in self.launches: k(grid, block, args)

    def breakdown(self, reps=10):
        """Milliseconds per token spent in each kind of kernel (each kind replayed alone as its own graph)."""
        out = {}
        with self.dev:
            names = {id(v): n for n, v in _K.items()}
            for kind in sorted({names[id(l[0])] for l in self.launches}):
                sub = [l for l in self.launches if names[id(l[0])] == kind]; s = cp.cuda.Stream(non_blocking=True)
                with s:
                    self.pos[...] = 0; s.synchronize(); s.begin_capture()
                    for k, grid, block, args in sub: k(grid, block, args)
                    g = s.end_capture(); best = 1e9
                    for _ in range(reps):
                        self.pos[...] = 0; s.synchronize(); t0 = time.perf_counter(); g.launch(s); s.synchronize(); best = min(best, time.perf_counter() - t0)
                out[kind] = dict(launches=len(sub), ms=round(best * 1000, 4))
            with self.stream: self.pos[...] = 0; self.stream.synchronize()
        return out

    def _feed(self, ids):
        self.pos[...] = 0
        for t in ids[:-1]:
            self.tok[...] = int(t); self.graph.launch(self.stream)
        self.tok[...] = int(ids[-1])

    def generate(self, ids, n):
        assert len(ids) + n <= self.L
        with self.dev, self.stream:
            self._feed(ids)
            for _ in range(n): self.graph.launch(self.stream)
            self.stream.synchronize()
            return cp.asnumpy(self.out[len(ids) - 1:len(ids) - 1 + n]).tolist()

    def first_logits(self, ids):
        """Logits after the last prompt token (for checking against a reference)."""
        with self.dev, self.stream:
            self._feed(ids); self.graph.launch(self.stream); self.stream.synchronize()
            return torch.as_tensor(cp.asnumpy(self.lg))

    def tok_per_s(self, ids, runs=5, new_tokens=64):
        best = 1e9
        with self.dev, self.stream:
            for _ in range(runs + 1):
                self._feed(ids); self.stream.synchronize(); t0 = time.perf_counter()
                for _ in range(new_tokens): self.graph.launch(self.stream)
                self.stream.synchronize(); best = min(best, time.perf_counter() - t0)
        return new_tokens / best
