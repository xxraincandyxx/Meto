from catalog.metal_shader import kernel


def solve(input, output, rows, cols):
    kernel(__file__)(input, output, rows, cols, threads=rows * cols)
