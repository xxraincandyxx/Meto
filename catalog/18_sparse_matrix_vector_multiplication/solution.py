"""Multiply a matrix containing zeros by a dense vector."""

import torch


def solve(A, x, y, M, N, nnz):
    # Here the interface supplies every entry in row-major order, including zeros.
    # Packing a sparse representation would read the entire matrix and allocate
    # indices for a single product. Use its existing strides for a native matvec.
    torch.mv(A.view(M, N), x, out=y)
