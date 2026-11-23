"""Apply the rectified linear activation to independent elements."""

from catalog.metal_shader import kernel


def solve(input, output, N):
    kernel(__file__, "rectify")(input, output, threads=N)
