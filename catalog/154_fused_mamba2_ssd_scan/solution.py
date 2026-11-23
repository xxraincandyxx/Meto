from catalog.metal_shader import kernel


def solve(x, dt, dt_bias, A, B, C, D, z, y, batch, seq_len, num_heads, head_dim,
          num_groups, state_dim):
    # Each (batch, head, channel) trajectory has an independent state vector.
    # Assign a SIMD group to it and keep up to four state entries per lane.
    kernel(__file__, "state_scan")(x, dt, dt_bias, A, B, C, D, z, y, seq_len,
                                   num_heads, head_dim, num_groups, state_dim,
                                   threads=batch * num_heads * head_dim * 32, group_size=32)
