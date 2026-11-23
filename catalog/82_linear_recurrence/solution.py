from catalog.metal_shader import kernel


def solve(a, x, h, B, L):
    kernel(__file__)(a, x, h, L, threads=B)
