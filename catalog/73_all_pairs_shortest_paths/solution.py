from catalog.metal_shader import kernel


def solve(dist, output, N):
    output.copy_(dist)
    relax = kernel(__file__)
    # Each dispatch completes its writes before the next pivot can consume them.
    for pivot in range(N):
        relax(output, N, pivot, threads=N * N)
