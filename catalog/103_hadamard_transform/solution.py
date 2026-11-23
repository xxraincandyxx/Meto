from catalog.metal_shader import kernel


def solve(input, output, B, N):
    # Each row fits in 16 KiB of threadgroup memory at the challenge limit.
    width = min(256, N)
    kernel(__file__, "butterflies")(input, output, N, threads=B * width, group_size=width)
