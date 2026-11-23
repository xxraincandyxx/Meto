from catalog.metal_shader import kernel


def solve(input, output, N):
    kernel(__file__)(input, output, threads=N)
