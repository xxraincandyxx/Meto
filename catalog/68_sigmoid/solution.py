from catalog.metal_shader import kernel


def solve(X, Y, N):
    kernel(__file__)(X, Y, threads=N)
