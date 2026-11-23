"""Count matches across a contiguous two-dimensional array."""

from importlib import import_module

count_flat = import_module("catalog.43_count_array_element.solution").solve


def solve(input, output, N, M, K):
    # Row-major storage places every row directly after the preceding one. The
    # count is independent of row boundaries, so all N*M elements can share the
    # one-dimensional SIMD reduction from lesson 43, with no reshape or copy.
    count_flat(input, output, N * M, K)
