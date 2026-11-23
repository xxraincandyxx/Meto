import torch

from catalog.metal_shader import kernel


def solve(A, B, result, N):
    # Each group handles 8 stripes of 256 neighboring pairs. Its eight
    # SIMD groups write one float partial each for the final GPU reduction.
    blocks = (N + 2047) // 2048
    partial = torch.empty(blocks * 8, dtype=torch.float32, device=A.device)
    kernel(__file__)(A, B, partial, N, threads=blocks * 256, group_size=256)
    result.copy_(partial.sum())
