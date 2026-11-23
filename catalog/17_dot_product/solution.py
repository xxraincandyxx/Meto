"""Reduce pairwise products to a scalar."""

import torch


def solve(A, B, result, N):
    # dot is a fused multiply/reduce operation. It avoids writing an N-element
    # array of products to device memory. A scalar view exposes the caller's
    # one-element output as the zero-dimensional destination dot expects.
    torch.dot(A, B, out=result.view(()))
