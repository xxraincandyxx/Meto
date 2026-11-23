from catalog.metal_shader import kernel


def solve(Q, cos, sin, output, M, D):
    kernel(__file__)(Q, cos, sin, output, D, threads=M * (D // 2))
