from catalog.metal_shader import kernel


def solve(
    Q,
    K_cache,
    V_cache,
    block_table,
    context_lens,
    output,
    batch_size,
    num_heads,
    head_dim,
    block_size,
    max_blocks_per_seq,
):
    # Scores are much smaller than the gathered key and value caches. Keep
    # the physical blocks in place and translate logical token coordinates
    # inside each kernel. No padding page is ever dereferenced.
    length = max_blocks_per_seq * block_size
    scores = Q.new_empty(batch_size, num_heads, length)
    kernel(__file__, "score")(
        Q,
        K_cache,
        block_table,
        context_lens,
        scores,
        num_heads,
        head_dim,
        block_size,
        max_blocks_per_seq,
        threads=scores.numel() * 32,
        group_size=32,
    )
    weights = scores.softmax(-1)
    kernel(__file__, "mix")(
        weights,
        V_cache,
        block_table,
        context_lens,
        output,
        num_heads,
        head_dim,
        block_size,
        max_blocks_per_seq,
        threads=output.numel(),
    )
