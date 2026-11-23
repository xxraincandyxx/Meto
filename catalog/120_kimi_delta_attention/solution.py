from catalog.metal_shader import kernel


def solve(Q, K, V, alpha, beta, output, batch, seq_len, d):
    # Independent columns of the recurrence each get a 32-lane SIMD group.
    kernel(__file__, "delta_rule")(Q, K, V, alpha, beta, output, seq_len, d,
                                   threads=batch * d * 32, group_size=32)
