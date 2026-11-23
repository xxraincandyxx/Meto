from catalog.metal_shader import kernel


def scan(a, x, h):
    batch, length = x.shape
    if length <= 32:
        # A short sequence fits cheaply in one lane; independent sequences
        # advance concurrently. No temporary buffers are needed.
        kernel(__file__)(a, x, h, length, threads=batch)
        return

    blocks = (length + 31) // 32
    gain = x.new_empty(batch, length)
    block_gain = x.new_empty(batch, blocks)
    block_bias = x.new_empty(batch, blocks)
    ends = x.new_empty(batch, blocks)
    kernel(__file__, "tiles")(a, x, h, gain, block_gain, block_bias, length, blocks,
                              threads=batch * blocks * 32, group_size=32)
    # Scan the summaries with the same associative operator. Each level has
    # 32 times fewer elements, so the recursion stays shallow for long rows.
    scan(block_gain, block_bias, ends)
    kernel(__file__, "propagate")(h, gain, ends, length, blocks, threads=batch * length)


def solve(a, x, h, B, L):
    scan(a, x, h)
