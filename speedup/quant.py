"""quant.py - integer storage formats for weight matrices. Weights are only rounded; nothing is retrained.

    i16   16-bit integers, one scale per row                                              16   bits per weight
    i8     8-bit integers, one scale per row                                               8   bits per weight
    n4     4-bit integers in blocks of 32 weights; one 16-bit scale code per block,
           one scale per row (the block rule is llama.cpp's Q4_0: the largest weight maps to -8)   4.5 bits per weight

quantize() returns the arrays the GPU kernels read (fused.py) and the exact float values they stand for, which is
what perplexity is measured on.
"""
import torch

BITS = {'i16': 16.0, 'i8': 8.0, 'n4': 4.5}
FORMATS = ['i16', 'i8', 'n4']                      # in order of fewer bits


def quantize(W, fmt):
    W = W.float()
    if fmt in ('i16', 'i8'):
        m = 32767.0 if fmt == 'i16' else 127.0
        s = (W.abs().amax(1) / m).clamp_min(1e-12)
        q = torch.clamp(torch.round(W / s[:, None]), -m, m)
        return dict(q=q.to(torch.int16 if fmt == 'i16' else torch.int8).contiguous(), scale=s.contiguous()), q * s[:, None]
    if fmt == 'n4':
        rows, n = W.shape; assert n % 32 == 0
        B = W.reshape(rows, n // 32, 32)
        m = torch.gather(B, 2, B.abs().argmax(2, keepdim=True)).squeeze(2)          # the signed largest weight of each block
        d = m / -8.0; d = torch.where(d == 0, torch.full_like(d, 1e-12), d)
        q = torch.clamp(torch.round(B / d[:, :, None]), -8, 7)
        D = d.abs().amax(1); code = torch.clamp(torch.round(d / D[:, None] * 32767.0), -32767, 32767)
        rs = D / 32767.0
        nib = (q + 8).to(torch.uint8).reshape(rows, n // 2, 2)
        packed = (nib[:, :, 0] | (nib[:, :, 1] << 4)).contiguous()                  # low nibble first
        deq = (q * (code * rs[:, None])[:, :, None]).reshape(rows, n)
        return dict(q=packed, code=code.to(torch.int16).contiguous(), scale=rs.contiguous()), deq
    raise ValueError(fmt)


MATS = ['q', 'k', 'v', 'o', 'g', 'u', 'dn']         # the weight matrices of a layer (norm weights stay in float)


def quantize_model(P, fmt_of, device):
    """P: smol.extract() weights. fmt_of(name) -> format, name = 'E' or (layer, matrix).
    Returns (arrays for the decoder, a copy of P holding the exact dequantized float weights)."""
    arrs = {}; Pq = dict(P); Pq['layers'] = []
    a, deq = quantize(P['E'].to(device), fmt_of('E')); arrs['E'] = (fmt_of('E'), a); Pq['E'] = deq
    for i, l in enumerate(P['layers']):
        nl = dict(l)
        for m in MATS:
            a, deq = quantize(l[m].to(device), fmt_of((i, m))); arrs[(i, m)] = (fmt_of((i, m)), a); nl[m] = deq
        Pq['layers'].append(nl)
    return arrs, Pq


def bytes_per_token(P, fmt_of):
    """Weight bytes read to produce one token (every matrix once; the embedding/output matrix once)."""
    b = P['E'].numel() * BITS[fmt_of('E')] / 8
    for i, l in enumerate(P['layers']):
        for m in MATS: b += l[m].numel() * BITS[fmt_of((i, m))] / 8
    return b
