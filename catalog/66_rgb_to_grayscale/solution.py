from catalog.metal_shader import kernel


def solve(input, output, width, height):
    kernel(__file__)(input, output, threads=width * height)
