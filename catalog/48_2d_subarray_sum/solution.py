"""Sum an inclusive rectangle without copying its rows."""

import torch


def solve(input, output, N, M, S_ROW, E_ROW, S_COL, E_COL):
    # A rectangle generally has gaps between rows in the original allocation.
    # Its strided view preserves these gaps. The device reduction follows the
    # strides and accumulates only the selected rows and columns.
    region = input[S_ROW:E_ROW + 1, S_COL:E_COL + 1]
    output.copy_(torch.sum(region, dtype=torch.int32))
