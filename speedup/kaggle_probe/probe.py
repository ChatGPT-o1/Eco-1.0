# probe.py - Kaggle T4: do custom integer-weight kernels run, are they exact, and how much of the GPU's memory speed do they reach?
import json, time, subprocess, sys
def say(**k): print(json.dumps(k), flush=True)
try:
    import cupy as cp
except Exception as e:
    say(event='pip', note=str(e)[:200]); subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda12x'], check=True); import cupy as cp
import numpy as np, torch
say(event='env', cupy=cp.__version__, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
    gpus=torch.cuda.device_count(), has_capture=hasattr(cp.cuda.Stream, 'begin_capture'))

SRC = r'''
#define LD4I(p) (*reinterpret_cast<const int4*>(p))
#define LD4F(p) (*reinterpret_cast<const float4*>(p))

extern "C" __global__ void noop(float* y) { if (threadIdx.x == 12345) y[0] = 0.f; }
extern "C" __global__ void inc(int* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1; }

// read-only streaming: each warp reads `per` runs of 512 contiguous bytes
extern "C" __global__ void stream_read(const int* __restrict__ a, float* __restrict__ out, const long long nchunks, const int per)
{
    const int lane = threadIdx.x & 31;
    const long long w = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const long long base = w * 32 * per;
    int acc = 0;
    for (int k = 0; k < per; k++) {
        const long long ch = base + (long long)k * 32 + lane;
        if (ch < nchunks) { const int4 v = LD4I(a + ch * 4); acc += v.x ^ v.y ^ v.z ^ v.w; }
    }
    if (acc == 0x7fffffff) out[0] = 1.f;
}

// y[r] (+)= scale[r] * sum_j W[r][j] * x[j].  One warp per row; each lane takes every 32nd 16-byte chunk of the row.
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
// 4-bit weights (nibble = q + 8, q in -8..7), 32 weights per 16-byte chunk, one signed 16-bit scale code per chunk
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

// h = x * rsqrt(mean(x^2) + eps) * w        (one block of n threads, n a multiple of 32, n <= 1024)
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
'''
K = {}
try:
    for name in ('noop', 'inc', 'stream_read', 'gemv_i16', 'gemv_i8', 'gemv_n4', 'prep'): K[name] = cp.RawKernel(SRC, name)
    K['noop']((1,), (32,), (cp.zeros(1, cp.float32),)); cp.cuda.Stream.null.synchronize(); say(event='compile', ok=True)
except Exception as e:
    say(event='compile', ok=False, error=str(e)[:3000]); raise

i32 = np.int32
def timed(launch, reps, runs=7):
    """Seconds per launch, measured inside a CUDA graph of `reps` launches (best of `runs` replays)."""
    s = cp.cuda.Stream(non_blocking=True)
    with s:
        launch(); s.synchronize(); s.begin_capture()
        for _ in range(reps): launch()
        g = s.end_capture(); best = 1e9
        for _ in range(runs):
            s.synchronize(); t0 = time.perf_counter(); g.launch(s); s.synchronize(); best = min(best, time.perf_counter() - t0)
    return best / reps

# ---- 1. graph capture keeps device-side state; cost of one launch
try:
    p = cp.zeros(1, cp.int32); s = cp.cuda.Stream(non_blocking=True)
    with s:
        K['inc']((1,), (32,), (p,)); s.synchronize(); p[...] = 0; s.synchronize(); s.begin_capture()
        for _ in range(3): K['inc']((1,), (32,), (p,))
        g = s.end_capture()
        for _ in range(5): g.launch(s)
        s.synchronize()
    y1 = cp.zeros(1, cp.float32)
    say(event='graph', state_after_5x3=int(p[0]), launch_overhead_us=round(1e6 * timed(lambda: K['noop']((1,), (32,), (y1,)), 2000), 3))
except Exception as e: say(event='graph', error=str(e)[:1500])

# ---- 2. how fast can this GPU read memory at all (read-only streaming of 512 MB)
try:
    a = cp.random.randint(-2**31, 2**31 - 1, size=128 * 1024 * 1024, dtype=cp.int32); nch = a.size // 4; out = cp.zeros(1, cp.float32); res = {}
    for per in (1, 2, 4, 8, 16):
        warps = -(-nch // (32 * per)); t = timed(lambda: K['stream_read'](((warps + 7) // 8,), (256,), (a, out, np.int64(nch), i32(per))), 3, runs=5)
        res[per] = round(a.nbytes / t / 1e9, 1)
    say(event='memory_read_GBps', loads_per_thread=res); del a
except Exception as e: say(event='memory_read_GBps', error=str(e)[:1500])

# ---- 3. quantizers (reference)
def q_i16(W):
    s = cp.maximum(cp.abs(W).max(axis=1) / 32767.0, 1e-12); q = cp.clip(cp.rint(W / s[:, None]), -32767, 32767).astype(cp.int16)
    return (q, s.astype(cp.float32)), q.astype(cp.float32) * s[:, None].astype(cp.float32), q.nbytes
def q_i8(W):
    s = cp.maximum(cp.abs(W).max(axis=1) / 127.0, 1e-12); q = cp.clip(cp.rint(W / s[:, None]), -127, 127).astype(cp.int8)
    return (q, s.astype(cp.float32)), q.astype(cp.float32) * s[:, None].astype(cp.float32), q.nbytes
def q_n4(W):
    rows, n = W.shape; B = W.reshape(rows, n // 32, 32)
    m = cp.take_along_axis(B, cp.abs(B).argmax(axis=2)[:, :, None], axis=2)[:, :, 0]
    d = m / -8.0; d = cp.where(d == 0, 1e-12, d)
    q = cp.clip(cp.rint(B / d[:, :, None]), -8, 7)
    D = cp.abs(d).max(axis=1); code = cp.clip(cp.rint(d / D[:, None] * 32767.0), -32767, 32767).astype(cp.int16)
    nib = (q + 8).astype(cp.uint8).reshape(rows, n // 2, 2); packed = cp.ascontiguousarray((nib[:, :, 0] | (nib[:, :, 1] << 4)).astype(cp.uint8))
    rs = (D / 32767.0).astype(cp.float32)
    deq = (q.astype(cp.float32) * (code.astype(cp.float32) * rs[:, None])[:, :, None]).reshape(rows, n)
    return (packed, code, rs), deq, packed.nbytes + code.nbytes
FMT = {'i16': (q_i16, 'gemv_i16'), 'i8': (q_i8, 'gemv_i8'), 'n4': (q_n4, 'gemv_n4')}

# ---- 4. exactness + speed of the gemv kernels on Smol's matrix shapes
shapes = [('q_or_o', 576, 576), ('k_or_v', 192, 576), ('qkv', 960, 576), ('gate_or_up', 1536, 576), ('gate+up', 3072, 576), ('down', 576, 1536), ('lm_head', 49152, 576)]
cp.random.seed(0)
for name, rows, n in shapes:
    W = cp.random.standard_normal((rows, n), dtype=cp.float32) * 0.05; x = cp.random.standard_normal(n, dtype=cp.float32); y = cp.zeros(rows, cp.float32)
    for fmt, (qf, kn) in FMT.items():
        try:
            arrs, deq, nbytes = qf(W); ref = deq @ x; args = arrs + (x, y, i32(rows), i32(n), i32(0))
            K[kn](((rows + 3) // 4,), (128,), args); cp.cuda.Stream.null.synchronize()
            err = float(cp.abs(y - ref).max() / (cp.abs(ref).max() + 1e-9))
            qerr = float(cp.sqrt(((deq - W) ** 2).mean()) / cp.sqrt((W ** 2).mean()))
            res = {}
            for rpb in (1, 2, 4, 8):
                t = timed(lambda: K[kn](((rows + rpb - 1) // rpb,), (32 * rpb,), args), 20 if rows > 10000 else 300)
                res[rpb] = dict(us=round(t * 1e6, 2), GBps=round(nbytes / t / 1e9, 1))
            say(event='gemv', matrix=name, rows=rows, n=n, fmt=fmt, MB=round(nbytes / 1e6, 3), max_rel_err_vs_reference=err, weight_rms_error=round(qerr, 5), rows_per_block=res)
        except Exception as e: say(event='gemv', matrix=name, fmt=fmt, error=str(e)[:1500])
    del W

# ---- 5. the norm kernel
try:
    x = cp.random.standard_normal(576, dtype=cp.float32) * 30; w = cp.random.standard_normal(576, dtype=cp.float32); h = cp.zeros(576, cp.float32)
    K['prep']((1,), (576,), (x, w, h, i32(576), np.float32(1e-5))); cp.cuda.Stream.null.synchronize()
    ref = x * (1.0 / cp.sqrt((x * x).mean() + 1e-5)) * w
    say(event='prep', max_rel_err=float(cp.abs(h - ref).max() / cp.abs(ref).max()), us=round(1e6 * timed(lambda: K['prep']((1,), (576,), (x, w, h, i32(576), np.float32(1e-5))), 1000), 3))
except Exception as e: say(event='prep', error=str(e)[:1500])

# ---- 6. what PyTorch's own matmul (cuBLAS) does on the same shapes, for reference
try:
    for name, rows, n in shapes:
        for dt, nm, bpw in ((torch.float32, 'fp32', 4), (torch.float16, 'fp16', 2)):
            Wt = torch.randn(rows, n, device='cuda', dtype=dt); h = torch.randn(1, n, device='cuda', dtype=dt); reps = 20 if rows > 10000 else 200
            def step():
                for _ in range(reps): o = h @ Wt.T
            step(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): step()
            best = 1e9
            for _ in range(7):
                torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize(); best = min(best, time.perf_counter() - t0)
            t = best / reps; say(event='torch_matmul', matrix=name, fmt=nm, us=round(t * 1e6, 2), GBps=round(rows * n * bpw / t / 1e9, 1))
            del g, Wt
except Exception as e: say(event='torch_matmul', error=str(e)[:1500])
say(event='done')
