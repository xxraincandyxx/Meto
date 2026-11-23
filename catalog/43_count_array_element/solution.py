"""Count matches using register counts and one atomic add per SIMD group."""

from catalog.metal_shader import kernel


def solve(input, output, N, K):
    output.zero_()
    groups = (N + 2047) // 2048
    kernel(__file__, "count_matches")(input, output, N, K, threads=groups * 256, group_size=256)
