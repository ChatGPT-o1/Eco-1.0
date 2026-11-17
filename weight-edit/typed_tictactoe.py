#!/usr/bin/env python3
"""typed_tictactoe.py - a small neural network that tells who has won a tic-tac-toe board. Every weight typed by hand.

    python typed_tictactoe.py         # checks the network on all 19,683 possible boards

Input: 18 numbers. Cells are numbered
    0 1 2
    3 4 5
    6 7 8
    inputs 0-8 = 1 where X is, inputs 9-17 = 1 where O is.
Hidden layer: 16 ReLU neurons, one per (player, line). The 8 lines:
    L0 top row  L1 middle row  L2 bottom row  L3 left column  L4 middle column  L5 right column  L6 diagonal 0-4-8  L7 diagonal 2-4-6
    Each neuron adds its line's 3 cells and subtracts 2, so it outputs 1 only when all 3 cells are that player's.
Output: 3 scores - X wins, O wins, nobody. Highest wins.
355 numbers in total.
"""
import itertools
import numpy as np

# 18 rows (inputs) x 16 columns (X lines L0..L7, then O lines L0..L7)
W1 = np.array([
    # X: L0 L1 L2 L3 L4 L5 L6 L7      O: L0 L1 L2 L3 L4 L5 L6 L7
    [1, 0, 0, 1, 0, 0, 1, 0,          0, 0, 0, 0, 0, 0, 0, 0],   # X in cell 0
    [1, 0, 0, 0, 1, 0, 0, 0,          0, 0, 0, 0, 0, 0, 0, 0],   # X in cell 1
    [1, 0, 0, 0, 0, 1, 0, 1,          0, 0, 0, 0, 0, 0, 0, 0],   # X in cell 2
    [0, 1, 0, 1, 0, 0, 0, 0,          0, 0, 0, 0, 0, 0, 0, 0],   # X in cell 3
    [0, 1, 0, 0, 1, 0, 1, 1,          0, 0, 0, 0, 0, 0, 0, 0],   # X in cell 4
    [0, 1, 0, 0, 0, 1, 0, 0,          0, 0, 0, 0, 0, 0, 0, 0],   # X in cell 5
    [0, 0, 1, 1, 0, 0, 0, 1,          0, 0, 0, 0, 0, 0, 0, 0],   # X in cell 6
    [0, 0, 1, 0, 1, 0, 0, 0,          0, 0, 0, 0, 0, 0, 0, 0],   # X in cell 7
    [0, 0, 1, 0, 0, 1, 1, 0,          0, 0, 0, 0, 0, 0, 0, 0],   # X in cell 8
    [0, 0, 0, 0, 0, 0, 0, 0,          1, 0, 0, 1, 0, 0, 1, 0],   # O in cell 0
    [0, 0, 0, 0, 0, 0, 0, 0,          1, 0, 0, 0, 1, 0, 0, 0],   # O in cell 1
    [0, 0, 0, 0, 0, 0, 0, 0,          1, 0, 0, 0, 0, 1, 0, 1],   # O in cell 2
    [0, 0, 0, 0, 0, 0, 0, 0,          0, 1, 0, 1, 0, 0, 0, 0],   # O in cell 3
    [0, 0, 0, 0, 0, 0, 0, 0,          0, 1, 0, 0, 1, 0, 1, 1],   # O in cell 4
    [0, 0, 0, 0, 0, 0, 0, 0,          0, 1, 0, 0, 0, 1, 0, 0],   # O in cell 5
    [0, 0, 0, 0, 0, 0, 0, 0,          0, 0, 1, 1, 0, 0, 0, 1],   # O in cell 6
    [0, 0, 0, 0, 0, 0, 0, 0,          0, 0, 1, 0, 1, 0, 0, 0],   # O in cell 7
    [0, 0, 0, 0, 0, 0, 0, 0,          0, 0, 1, 0, 0, 1, 1, 0],   # O in cell 8
], dtype=float)
b1 = np.array([-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2], dtype=float)

# 16 rows (hidden neurons) x 3 columns (X wins, O wins, nobody)
W2 = np.array([
    [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],   # X lines
    [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],   # O lines
], dtype=float)
b2 = np.array([0, 0, 0.5])

def network(board):
    """board: 9 characters from 'X', 'O', '.'"""
    x = np.array([c == 'X' for c in board] + [c == 'O' for c in board], dtype=float)
    return ['X wins', 'O wins', 'nobody'][int(np.argmax(np.maximum(0, x @ W1 + b1) @ W2 + b2))]

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
def rule(board):
    x = any(all(board[i] == 'X' for i in l) for l in LINES); o = any(all(board[i] == 'O' for i in l) for l in LINES)
    return 'both' if x and o else 'X wins' if x else 'O wins' if o else 'nobody'

if __name__ == '__main__':
    right = total = 0
    for cells in itertools.product('XO.', repeat=9):
        b = ''.join(cells); r = rule(b)
        if r == 'both': continue                      # impossible in a real game
        total += 1; right += network(b) == r
    print(f'355 typed numbers. Correct on {right:,} of {total:,} boards ({100 * right / total:.2f}%).')
    for b in ('XXX.O.O..', 'OX.XO.X.O', 'XOXOXOOXO', '.........'):
        print(b[:3], b[3:6], b[6:], '->', network(b))
