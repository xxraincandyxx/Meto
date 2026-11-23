from catalog.metal_shader import kernel


def solve(X, S, Y, M, N, TILE_SIZE):
    kernel(__file__)(X, S, Y, N, TILE_SIZE, threads=M * N)
