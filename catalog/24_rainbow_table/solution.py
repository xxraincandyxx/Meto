"""Compute repeated four-byte FNV-1a checksums in registers."""

from catalog.metal_shader import kernel


def solve(input, output, N, R):
    kernel(__file__, "hash_words")(input, output, R, threads=N)
