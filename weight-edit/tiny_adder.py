#!/usr/bin/env python3
"""tiny_adder.py - a ~1K-parameter neural network whose weights are written by hand, compared with the same network trained.

Task: add two 6-bit numbers (0-63 each) given as bits; output the 7 bits of the sum. All 4,096 pairs are tested.

Architecture (residual MLP, like one transformer MLP block repeated):
    state x has 20 slots: a0..a5 | b0..b5 | s0..s6 (sum bits) | c (carry)
    6 stages, each: x <- x + U @ relu(V @ x + bias)        V: 4x20, bias: 4, U: 20x4   -> 164 params per stage
    total 6 x 164 = 984 parameters

Hand-written weights (a ripple-carry adder). At stage k let t = a_k + b_k + c (0..3):
    n0 = relu(t)        = t
    n1 = relu(t - 1)
    n2 = relu(t - 2)
    n3 = relu(c)        = c
    new carry = n1 - n2          (1 when t >= 2)
    sum bit   = t - 2*new carry = n0 - 2*n1 + 2*n2
    so: s_k += n0 - 2*n1 + 2*n2,   c += n1 - n2 - n3   (replace the old carry with the new one)
After the last stage the carry is the top sum bit s6.

    python tiny_adder.py            # hand-written vs trained (512, 2048 labelled examples)
"""
import itertools, time
import torch

BITS = 6; A, B, S, C = 0, BITS, 2 * BITS, 3 * BITS + 1; WIDTH = 3 * BITS + 2   # slots: a | b | s (7) | c
STAGES, HID = BITS, 4

def encode(a, b):
    x = torch.zeros(WIDTH)
    for k in range(BITS): x[A + k] = (a >> k) & 1; x[B + k] = (b >> k) & 1
    return x

ALL = [(a, b) for a in range(2 ** BITS) for b in range(2 ** BITS)]
X = torch.stack([encode(a, b) for a, b in ALL])
Y = torch.tensor([[(a + b) >> k & 1 for k in range(BITS + 1)] for a, b in ALL], dtype=torch.float32)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.V = torch.nn.Parameter(torch.randn(STAGES, HID, WIDTH) * 0.3)
        self.bias = torch.nn.Parameter(torch.zeros(STAGES, HID))
        self.U = torch.nn.Parameter(torch.randn(STAGES, WIDTH, HID) * 0.3)
    def forward(self, x):
        for k in range(STAGES):
            x = x + torch.relu(x @ self.V[k].T + self.bias[k]) @ self.U[k].T
        return x[:, S:S + BITS + 1]          # sum bits (the carry slot is copied into s6 by the last stage)
    def params(self): return sum(p.numel() for p in self.parameters())

def hand_written():
    net = Net()
    with torch.no_grad():
        net.V.zero_(); net.bias.zero_(); net.U.zero_()
        for k in range(STAGES):
            for n in range(3):                                  # n0, n1, n2 all read t = a_k + b_k + c
                net.V[k, n, A + k] = 1; net.V[k, n, B + k] = 1; net.V[k, n, C] = 1
            net.bias[k, 1] = -1; net.bias[k, 2] = -2
            net.V[k, 3, C] = 1                                  # n3 = c
            net.U[k, S + k, 0] = 1; net.U[k, S + k, 1] = -2; net.U[k, S + k, 2] = 2      # sum bit
            if k < STAGES - 1:
                net.U[k, C, 1] = 1; net.U[k, C, 2] = -1; net.U[k, C, 3] = -1             # carry <- new carry
            else:
                net.U[k, S + BITS, 1] = 1; net.U[k, S + BITS, 2] = -1                     # last carry is the top sum bit
    return net

def accuracy(net):
    with torch.no_grad():
        pred = (net(X) > 0.5).float()
    return float((pred == Y).all(1).float().mean())

def trained(n_examples, steps=4000, seed=0):
    torch.manual_seed(seed); net = Net()
    idx = torch.randperm(len(ALL))[:n_examples]
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad(); loss = torch.nn.functional.mse_loss(net(X[idx]), Y[idx]); loss.backward(); opt.step()
    return net

if __name__ == '__main__':
    rows = []
    h = hand_written()
    rows.append(('Hand-written', 0, int(sum((p != 0).sum() for p in h.parameters())), accuracy(h)))
    for n in (512, 2048):
        t0 = time.time(); t = trained(n)
        rows.append((f'Trained ({time.time() - t0:.0f}s)', n, int(sum((p != 0).sum() for p in t.parameters())), accuracy(t)))
    print(f'network size: {Net().params()} parameters; test: all {len(ALL)} additions of two {BITS}-bit numbers\n')
    print(f'{"Weights":20s} {"Labels seen":>12s} {"Non-zero params":>16s} {"Exact answers":>14s}')
    for name, n, nz, acc in rows: print(f'{name:20s} {n:12d} {nz:16d} {acc * 100:13.1f}%')
