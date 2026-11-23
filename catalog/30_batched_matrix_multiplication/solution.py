"""Multiply independent pairs of matrices in one batched GPU operation."""

import torch


def solve(A, B, C, BATCH, M, N, K):
    # The first stride identifies a matrix pair. A batched dispatch avoids
    # paying a Python/GPU dispatch cost for every matrix separately. The final
    # two axes identify the contracting dimension K and the output axes M and N.
    torch.bmm(A.view(BATCH, M, K), B.view(BATCH, K, N), out=C.view(BATCH, M, N))
