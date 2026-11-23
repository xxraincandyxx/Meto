"""Add two contiguous square matrices in four-float packets."""

from catalog.metal_shader import kernel


def solve(A, B, C, N):
    count = N * N
    kernel(__file__, "add")(A, B, C, count, threads=(count + 3) // 4)
