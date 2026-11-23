"""Find every point’s nearest other point with a tiled GPU search."""

from catalog.metal_shader import kernel


def solve(points, indices, N):
    kernel(__file__, "nearest")(points, indices, N, threads=((N + 127) // 128) * 128, group_size=128)
