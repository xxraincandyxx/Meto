"""Compute a leaky rectified linear activation in a single Metal pass."""

from catalog.metal_shader import kernel


def solve(input, output, N):
    kernel(__file__, "activate")(input, output, threads=N)
