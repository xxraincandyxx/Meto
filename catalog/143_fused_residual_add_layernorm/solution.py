from catalog.metal_shader import kernel


def solve(x, residual, gamma, beta, y, residual_out, mean, rstd, N, C, eps):
    # One SIMD group normalizes a row. Its reduction primitives broadcast
    # the same statistics to all 32 lanes without a threadgroup scratch array.
    kernel(__file__, "normalize")(x, residual, gamma, beta, y, residual_out,
                                  mean, rstd, C, eps, threads=N * 32, group_size=32)
