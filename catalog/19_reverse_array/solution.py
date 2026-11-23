"""Reverse a float array in place with disjoint pair swaps."""

from catalog.metal_shader import kernel


def solve(input, N):
    if N >= 2:
        kernel(__file__, "reverse_pairs")(input, N, threads=N // 2)
