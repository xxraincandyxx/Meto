"""Compute an integer matrix power by repeated squaring."""

import torch


def solve(input, output, N, P):
    # Binary exponentiation reduces the number of GEMMs to O(log P). Each set
    # bit selects a power A, A^2, A^4, ... that belongs in the product. Delay
    # the first multiply until two factors exist to avoid multiplying by identity.
    factor = input.view(N, N)
    accumulated = None
    while P:
        if P & 1:
            accumulated = factor if accumulated is None else accumulated @ factor
        P >>= 1
        if P:
            factor = factor @ factor
    output.copy_(accumulated.flatten())
