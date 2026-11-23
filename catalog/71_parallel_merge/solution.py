from catalog.metal_shader import kernel


def solve(A, B, C, M, N):
    kernel(__file__)(A, B, C, M, N, threads=M + N)
