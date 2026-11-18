# probe.py - Kaggle T4: which integer-weight kernel design gets closest to the GPU's memory speed?
#   v1  one warp per row; x read from memory for every chunk
#   v2  each warp walks many rows; each lane keeps its slice of x in registers (float math)
#   v3  like v2 with x as 16-bit integers and integer multiply-add
import json, time, subprocess, sys
def say(**k): print(json.dumps(k), flush=True)
try:
    import cupy as cp
except Exception as e:
    say(event='pip', note=str(e)[:200]); subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda12x'], check=True); import cupy as cp
import numpy as np, torch
say(event='env', cupy=cp.__version__, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))

SRC = r'''
#define LD4I(p) (*reinterpret_cast<const int4*>(p))
#define LD4F(p) (*reinterpret_cast<const float4*>(p))
#ifndef N
#define N 576
#endif

extern "C" __global__ void noop(float* y) { if (threadIdx.x == 12345) y[0] = 0.f; }
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

// ======================================================================== v1 (as in the first decoder)
extern "C" __global__ void gemv1_i16(const short* __restrict__ W, const float* __restrict__ scale, const float* __restrict__ x,
                                     float* __restrict__ y, const int rows, const int n, const int add)
{
    const int lane = threadIdx.x & 31; const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const short* row = W + (size_t)r * n; float acc = 0.f;
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
extern "C" __global__ void gemv1_i8(const signed char* __restrict__ W, const float* __restrict__ scale, const float* __restrict__ x,
                                    float* __restrict__ y, const int rows, const int n, const int add)
{
    const int lane = threadIdx.x & 31; const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const signed char* row = W + (size_t)r * n; float acc = 0.f;
    for (int c = lane * 16; c < n; c += 512) {
        const int4 w = LD4I(row + c);
        acc += dot4b(w.x, LD4F(x + c)) + dot4b(w.y, LD4F(x + c + 4)) + dot4b(w.z, LD4F(x + c + 8)) + dot4b(w.w, LD4F(x + c + 12));
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) { const float v = acc * scale[r]; y[r] = add ? y[r] + v : v; }
}

// ======================================================================== v2: warps walk rows, x in registers. N is a compile-time constant.
#define ROWS_PREAMBLE \
    const int lane = threadIdx.x & 31; \
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * rpw; \
    if (r0 >= rows) return; \
    const int r1 = min(r0 + rpw, rows);
#define ROW_END \
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o); \
    if (lane == 0) { const float v = acc * scale[r]; y[r] = add ? y[r] + v : v; }

// ---- 16-bit integers: unit = 8 weights = 16 bytes
#define U16 (N / 8)
#define IT16 ((U16 + 31) / 32)
extern "C" __global__ void gemv2_i16(const short* __restrict__ W, const float* __restrict__ scale, const float* __restrict__ x,
                                     float* __restrict__ y, const int rows, const int rpw, const int add)
{
    ROWS_PREAMBLE
    float xr[IT16][8];
    #pragma unroll
    for (int k = 0; k < IT16; k++) {
        const int u = lane + 32 * k;
        #pragma unroll
        for (int j = 0; j < 8; j++) xr[k][j] = (u < U16) ? x[u * 8 + j] : 0.f;
    }
    for (int r = r0; r < r1; r++) {
        const short* row = W + (size_t)r * N; float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < IT16; k++) {
            const int u = lane + 32 * k;
            if (u < U16) {
                const int4 w = LD4I(row + u * 8);
                acc += (float)(short)(w.x) * xr[k][0] + (float)(w.x >> 16) * xr[k][1] + (float)(short)(w.y) * xr[k][2] + (float)(w.y >> 16) * xr[k][3]
                     + (float)(short)(w.z) * xr[k][4] + (float)(w.z >> 16) * xr[k][5] + (float)(short)(w.w) * xr[k][6] + (float)(w.w >> 16) * xr[k][7];
            }
        }
        ROW_END
    }
}

// ---- 8-bit integers: unit = 16 weights = 16 bytes
#define U8 (N / 16)
#define IT8 ((U8 + 31) / 32)
#define D4(w, a, b, c, d) ((float)(signed char)(w) * (a) + (float)(signed char)((w) >> 8) * (b) + (float)(signed char)((w) >> 16) * (c) + (float)((w) >> 24) * (d))
extern "C" __global__ void gemv2_i8(const signed char* __restrict__ W, const float* __restrict__ scale, const float* __restrict__ x,
                                    float* __restrict__ y, const int rows, const int rpw, const int add)
{
    ROWS_PREAMBLE
    float xr[IT8][16];
    #pragma unroll
    for (int k = 0; k < IT8; k++) {
        const int u = lane + 32 * k;
        #pragma unroll
        for (int j = 0; j < 16; j++) xr[k][j] = (u < U8) ? x[u * 16 + j] : 0.f;
    }
    for (int r = r0; r < r1; r++) {
        const signed char* row = W + (size_t)r * N; float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < IT8; k++) {
            const int u = lane + 32 * k;
            if (u < U8) {
                const int4 w = LD4I(row + u * 16);
                acc += D4(w.x, xr[k][0], xr[k][1], xr[k][2], xr[k][3]) + D4(w.y, xr[k][4], xr[k][5], xr[k][6], xr[k][7])
                     + D4(w.z, xr[k][8], xr[k][9], xr[k][10], xr[k][11]) + D4(w.w, xr[k][12], xr[k][13], xr[k][14], xr[k][15]);
            }
        }
        ROW_END
    }
}
#define I4(w, a, b, c, d) ((int)(signed char)(w) * (a) + (int)(signed char)((w) >> 8) * (b) + (int)(signed char)((w) >> 16) * (c) + ((w) >> 24) * (d))
extern "C" __global__ void gemv3_i8(const signed char* __restrict__ W, const float* __restrict__ scale, const int* __restrict__ xq, const float* __restrict__ xs,
                                    float* __restrict__ y, const int rows, const int rpw, const int add)
{
    ROWS_PREAMBLE
    int xr[IT8][16];
    #pragma unroll
    for (int k = 0; k < IT8; k++) {
        const int u = lane + 32 * k;
        #pragma unroll
        for (int j = 0; j < 16; j++) xr[k][j] = (u < U8) ? xq[u * 16 + j] : 0;
    }
    const float xsc = xs[0];
    for (int r = r0; r < r1; r++) {
        const signed char* row = W + (size_t)r * N; float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < IT8; k++) {
            const int u = lane + 32 * k;
            if (u < U8) {
                const int4 w = LD4I(row + u * 16);
                acc += (float)(I4(w.x, xr[k][0], xr[k][1], xr[k][2], xr[k][3]) + I4(w.y, xr[k][4], xr[k][5], xr[k][6], xr[k][7])
                             + I4(w.z, xr[k][8], xr[k][9], xr[k][10], xr[k][11]) + I4(w.w, xr[k][12], xr[k][13], xr[k][14], xr[k][15]));
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) { const float v = acc * scale[r] * xsc; y[r] = add ? y[r] + v : v; }
    }
}

// ---- B-bit integers in blocks of 32 weights (B words), one 16-bit scale code per block. Unit = 16 weights = B/2 words.
#define UN (N / 16)
#define ITN ((UN + 31) / 32)
#define NIB8(w, x, o) ((float)((w) & 15u) * x[o] + (float)(((w) >> 4) & 15u) * x[o + 1] + (float)(((w) >> 8) & 15u) * x[o + 2] + (float)(((w) >> 12) & 15u) * x[o + 3] \
                     + (float)(((w) >> 16) & 15u) * x[o + 4] + (float)(((w) >> 20) & 15u) * x[o + 5] + (float)(((w) >> 24) & 15u) * x[o + 6] + (float)((w) >> 28) * x[o + 7])
extern "C" __global__ void gemv2_n4(const unsigned int* __restrict__ W, const short* __restrict__ bs, const float* __restrict__ scale,
                                    const float* __restrict__ x, float* __restrict__ y, const int rows, const int rpw, const int add)
{
    ROWS_PREAMBLE
    float xr[ITN][16], sx[ITN];
    #pragma unroll
    for (int k = 0; k < ITN; k++) {
        const int u = lane + 32 * k; float s = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; j++) { xr[k][j] = (u < UN) ? x[u * 16 + j] : 0.f; s += xr[k][j]; }
        sx[k] = 8.f * s;
    }
    for (int r = r0; r < r1; r++) {
        const unsigned int* row = W + (size_t)r * (N / 8); const short* bsr = bs + (size_t)r * (N / 32); float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < ITN; k++) {
            const int u = lane + 32 * k;
            if (u < UN) {
                const unsigned int w0 = row[u * 2], w1 = row[u * 2 + 1];
                acc += (float)bsr[u >> 1] * (NIB8(w0, xr[k], 0) + NIB8(w1, xr[k], 8) - sx[k]);
            }
        }
        ROW_END
    }
}
#define INIB8(w, x, o) ((int)((w) & 15u) * x[o] + (int)(((w) >> 4) & 15u) * x[o + 1] + (int)(((w) >> 8) & 15u) * x[o + 2] + (int)(((w) >> 12) & 15u) * x[o + 3] \
                      + (int)(((w) >> 16) & 15u) * x[o + 4] + (int)(((w) >> 20) & 15u) * x[o + 5] + (int)(((w) >> 24) & 15u) * x[o + 6] + (int)((w) >> 28) * x[o + 7])
extern "C" __global__ void gemv3_n4(const unsigned int* __restrict__ W, const short* __restrict__ bs, const float* __restrict__ scale,
                                    const int* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ y, const int rows, const int rpw, const int add)
{
    ROWS_PREAMBLE
    int xr[ITN][16], sx[ITN];
    #pragma unroll
    for (int k = 0; k < ITN; k++) {
        const int u = lane + 32 * k; int s = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) { xr[k][j] = (u < UN) ? xq[u * 16 + j] : 0; s += xr[k][j]; }
        sx[k] = 8 * s;
    }
    const float xsc = xs[0];
    for (int r = r0; r < r1; r++) {
        const unsigned int* row = W + (size_t)r * (N / 8); const short* bsr = bs + (size_t)r * (N / 32); float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < ITN; k++) {
            const int u = lane + 32 * k;
            if (u < UN) {
                const unsigned int w0 = row[u * 2], w1 = row[u * 2 + 1];
                acc += (float)bsr[u >> 1] * (float)(INIB8(w0, xr[k], 0) + INIB8(w1, xr[k], 8) - sx[k]);
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) { const float v = acc * scale[r] * xsc; y[r] = add ? y[r] + v : v; }
    }
}
// 6-bit: 16 weights = 96 bits = 3 words; value = field - 32
#define F6(v) ((float)((v) & 63u))
extern "C" __global__ void gemv2_n6(const unsigned int* __restrict__ W, const short* __restrict__ bs, const float* __restrict__ scale,
                                    const float* __restrict__ x, float* __restrict__ y, const int rows, const int rpw, const int add)
{
    ROWS_PREAMBLE
    float xr[ITN][16], sx[ITN];
    #pragma unroll
    for (int k = 0; k < ITN; k++) {
        const int u = lane + 32 * k; float s = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; j++) { xr[k][j] = (u < UN) ? x[u * 16 + j] : 0.f; s += xr[k][j]; }
        sx[k] = 32.f * s;
    }
    for (int r = r0; r < r1; r++) {
        const unsigned int* row = W + (size_t)r * (N / 32 * 6); const short* bsr = bs + (size_t)r * (N / 32); float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < ITN; k++) {
            const int u = lane + 32 * k;
            if (u < UN) {
                const unsigned int a = row[u * 3], b = row[u * 3 + 1], c = row[u * 3 + 2]; const float* xx = xr[k];
                const float s = F6(a) * xx[0] + F6(a >> 6) * xx[1] + F6(a >> 12) * xx[2] + F6(a >> 18) * xx[3] + F6(a >> 24) * xx[4]
                              + F6((a >> 30) | (b << 2)) * xx[5] + F6(b >> 4) * xx[6] + F6(b >> 10) * xx[7] + F6(b >> 16) * xx[8] + F6(b >> 22) * xx[9]
                              + F6((b >> 28) | (c << 4)) * xx[10] + F6(c >> 2) * xx[11] + F6(c >> 8) * xx[12] + F6(c >> 14) * xx[13] + F6(c >> 20) * xx[14] + F6(c >> 26) * xx[15];
                acc += (float)bsr[u >> 1] * (s - sx[k]);
            }
        }
        ROW_END
    }
}
'''
i32 = np.int32
_kc = {}
def K(name, n=576):
    if (name, n) not in _kc: _kc[(name, n)] = cp.RawKernel(SRC, name, options=(f'-DN={n}',))
    return _kc[(name, n)]
def timed(launch, reps, runs=7):
    s = cp.cuda.Stream(non_blocking=True)
    with s:
        launch(); s.synchronize(); s.begin_capture()
        for _ in range(reps): launch()
        g = s.end_capture(); best = 1e9
        for _ in range(runs):
            s.synchronize(); t0 = time.perf_counter(); g.launch(s); s.synchronize(); best = min(best, time.perf_counter() - t0)
    return best / reps
try:
    y1 = cp.zeros(1, cp.float32); K('noop')((1,), (32,), (y1,)); cp.cuda.Stream.null.synchronize()
    say(event='compile', ok=True, launch_overhead_us=round(1e6 * timed(lambda: K('noop')((1,), (32,), (y1,)), 2000), 3))
except Exception as e:
    say(event='compile', ok=False, error=str(e)[:3000]); raise
try:
    a = cp.random.randint(-2**31, 2**31 - 1, size=128 * 1024 * 1024, dtype=cp.int32); nch = a.size // 4; out = cp.zeros(1, cp.float32); res = {}
    for per in (1, 4, 16, 64):
        warps = -(-nch // (32 * per)); t = timed(lambda: K('stream_read')(((warps + 7) // 8,), (256,), (a, out, np.int64(nch), i32(per))), 3, runs=5)
        res[per] = round(a.nbytes / t / 1e9, 1)
    say(event='memory_read_GBps', loads_per_thread=res); del a
except Exception as e: say(event='memory_read_GBps', error=str(e)[:1500])

def q_row(W, m, dt):
    s = cp.maximum(cp.abs(W).max(axis=1) / m, 1e-12); q = cp.clip(cp.rint(W / s[:, None]), -m, m).astype(dt)
    return (q, s.astype(cp.float32)), q.astype(cp.float32) * s[:, None].astype(cp.float32), q.nbytes
def q_nb(W, B):
    rows, n = W.shape; Bk = W.reshape(rows, n // 32, 32); lo = -(1 << (B - 1)); hi = -lo - 1
    m = cp.take_along_axis(Bk, cp.abs(Bk).argmax(axis=2)[:, :, None], axis=2)[:, :, 0]
    d = m / lo; d = cp.where(d == 0, 1e-12, d)
    q = cp.clip(cp.rint(Bk / d[:, :, None]), lo, hi)
    D = cp.abs(d).max(axis=1); code = cp.clip(cp.rint(d / D[:, None] * 32767.0), -32767, 32767).astype(cp.int16)
    f = (q - lo).astype(cp.uint64); words = cp.zeros((rows, n // 32, B), cp.uint64)
    for j in range(32):
        o = j * B; wi, sh = o >> 5, o & 31
        words[:, :, wi] |= (f[:, :, j] << cp.uint64(sh)) & cp.uint64(0xffffffff)
        if sh + B > 32: words[:, :, wi + 1] |= f[:, :, j] >> cp.uint64(32 - sh)
    packed = cp.ascontiguousarray(words.astype(cp.uint32).reshape(rows, -1)); rs = (D / 32767.0).astype(cp.float32)
    deq = (q.astype(cp.float32) * (code.astype(cp.float32) * rs[:, None])[:, :, None]).reshape(rows, n)
    return (packed, code, rs), deq, packed.nbytes + code.nbytes
QF = {'i16': lambda W: q_row(W, 32767.0, cp.int16), 'i8': lambda W: q_row(W, 127.0, cp.int8), 'n6': lambda W: q_nb(W, 6), 'n4': lambda W: q_nb(W, 4)}

shapes = [('k_or_v', 192, 576), ('q_or_o', 576, 576), ('qkv', 960, 576), ('gate_or_up', 1536, 576), ('gate+up', 3072, 576), ('down', 576, 1536), ('lm_head', 49152, 576)]
cp.random.seed(0)
for name, rows, n in shapes:
    W = cp.random.standard_normal((rows, n), dtype=cp.float32) * 0.05; x = cp.random.standard_normal(n, dtype=cp.float32); y = cp.zeros(rows, cp.float32)
    xsv = float(cp.abs(x).max()) / 32767.0; xq = cp.rint(x / xsv).astype(cp.int32); xs = cp.asarray([xsv], dtype=cp.float32); xdq = xq.astype(cp.float32) * xs[0]
    reps = 20 if rows > 10000 else 300
    for fmt, qf in QF.items():
        out = dict(event='gemv', matrix=name, rows=rows, n=n, fmt=fmt)
        try:
            arrs, deq, nbytes = qf(W); ref = deq @ x; refq = deq @ xdq; out['MB'] = round(nbytes / 1e6, 3)
            def run(kname, args_fn, sweep, refv, v1=False):
                res = {}; best = None
                for p in sweep:
                    if v1: grid, block, args = ((rows + p - 1) // p,), (32 * p,), args_fn(n)
                    else:
                        if rows / p < 4: continue
                        grid, block, args = (-(-rows // p),), (32,), args_fn(p)
                    k = K(kname, n); y[...] = 0; k(grid, block, args); cp.cuda.Stream.null.synchronize()
                    err = float(cp.abs(y - refv).max() / (cp.abs(refv).max() + 1e-9))
                    t = timed(lambda: k(grid, block, args), reps)
                    res[p] = [round(t * 1e6, 2), round(nbytes / t / 1e9, 1)]
                    if err > 1e-3: res[p].append(f'WRONG err={err:.3g}')
                    if best is None or t < best[1]: best = (p, t)
                return dict(best=dict(param=best[0], us=round(best[1] * 1e6, 2), GBps=round(nbytes / best[1] / 1e9, 1), Gweights_per_s=round(rows * n / best[1] / 1e9, 1)), all=res)
            sw = (1, 2, 4, 8, 16, 32, 64, 128)
            if fmt in ('i16', 'i8'):
                out['v1'] = run('gemv1_' + fmt, lambda nn: arrs + (x, y, i32(rows), i32(nn), i32(0)), (4,), ref, v1=True)
                out['v2'] = run('gemv2_' + fmt, lambda p: arrs + (x, y, i32(rows), i32(p), i32(0)), sw, ref)
                if fmt == 'i8': out['v3'] = run('gemv3_i8', lambda p: arrs + (xq, xs, y, i32(rows), i32(p), i32(0)), sw, refq)
            else:
                out['v2'] = run('gemv2_' + fmt, lambda p: arrs + (x, y, i32(rows), i32(p), i32(0)), sw, ref)
                if fmt == 'n4': out['v3'] = run('gemv3_n4', lambda p: arrs + (xq, xs, y, i32(rows), i32(p), i32(0)), sw, refq)
        except Exception as e: out['error'] = str(e)[:2500]
        say(**out)
    del W
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
            t = best / reps; say(event='torch_matmul', matrix=name, fmt=nm, us=round(t * 1e6, 2), GBps=round(rows * n * bpw / t / 1e9, 1), Gweights_per_s=round(rows * n / t / 1e9, 1))
            del g, Wt
except Exception as e: say(event='torch_matmul', error=str(e)[:1500])
say(event='done')
