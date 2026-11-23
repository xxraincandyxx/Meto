from catalog.metal_shader import kernel


def solve(Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim):
    # Int8 scales are constant across each token's features. Fuse the key scale
    # into the score and value scale into the final weight. Only H*S scores
    # need float scratch; the two H*S*D caches stay in their compact format.
    scores = Q.new_empty(num_heads, seq_len)
    kernel(__file__, "score")(
        Q, K_int8, k_scale, scores, seq_len, head_dim, threads=scores.numel() * 32, group_size=32
    )
    weights = scores.softmax(-1) * v_scale
    kernel(__file__, "mix")(weights, V_int8, output, seq_len, head_dim, threads=output.numel())
