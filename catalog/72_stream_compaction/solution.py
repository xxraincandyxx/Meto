import torch
from catalog.metal_shader import kernel


def solve(A, N, out):
    counts = torch.cumsum(A > 0, 0, dtype=torch.int32)
    kernel(__file__)(A, counts, out, threads=N)
