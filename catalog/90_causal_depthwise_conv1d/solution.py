from catalog.metal_shader import kernel


def solve(x, weight, bias, output, B, L, D, K):
    kernel(__file__)(x, weight, bias, output, L, D, K, threads=B * L * D)
