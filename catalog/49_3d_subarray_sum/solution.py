"""Reduce the selected inclusive box inside a three-dimensional tensor."""

import torch


def solve(input, output, N, M, K, S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL):
    # All six coordinates are inclusive. The view carries depth and row strides,
    # allowing the GPU reduction to skip intervening storage outside the box.
    box = input[S_DEP:E_DEP + 1, S_ROW:E_ROW + 1, S_COL:E_COL + 1]
    output.copy_(torch.sum(box, dtype=torch.int32))
