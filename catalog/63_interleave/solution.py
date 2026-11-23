from catalog.metal_shader import kernel


def solve(A, B, output, N):
    kernel(__file__)(A, B, output, threads=N)
