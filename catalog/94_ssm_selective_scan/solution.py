from catalog.metal_shader import kernel


def solve(u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state):
    # One Apple SIMD group owns a batch/channel pair. Its 32 lanes distribute
    # up to 64 state coordinates; recurrence stays in registers across steps.
    kernel(__file__)(
        u,
        delta,
        A,
        B,
        C,
        skip,
        y,
        seq_len,
        d_model,
        d_state,
        threads=batch * d_model * 32,
        group_size=32,
    )
