from catalog.metal_shader import kernel


def solve(x, residual, weight, out, N, C, eps):
    # Each row normalizes independently. Launch one Apple SIMD group for
    # each row so neighboring lanes can share their partial sums directly.
    kernel(__file__)(x, residual, weight, out, C, eps, threads=N * 32, group_size=32)
