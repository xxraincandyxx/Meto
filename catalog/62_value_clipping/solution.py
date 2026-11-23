from catalog.metal_shader import kernel


def solve(input, output, lo, hi, N):
    kernel(__file__)(input, output, float(lo), float(hi), threads=N)
