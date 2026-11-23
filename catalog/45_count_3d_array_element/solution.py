"""Count matching voxels throughout a contiguous volume."""

from importlib import import_module

count_flat = import_module("catalog.43_count_array_element.solution").solve


def solve(input, output, N, M, K, P):
    # Each depth plane contributes M*K consecutive integers. Summing over the
    # whole volume can use the same linear GPU grid and register-local match
    # counts as lesson 43. P is the target value and K is the innermost extent.
    count_flat(input, output, N * M * K, P)
