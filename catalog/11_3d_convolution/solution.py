"""Evaluate valid three-dimensional cross-correlation."""

from catalog.metal_shader import kernel as metal_kernel


def solve(input, kernel, output, input_depth, input_rows, input_cols, kernel_depth, kernel_rows, kernel_cols):
    grid = [input_cols - kernel_cols + 1, input_rows - kernel_rows + 1, input_depth - kernel_depth + 1]
    metal_kernel(__file__, "filter_volume")(input, kernel, output, input_rows, input_cols,
                            kernel_depth, kernel_rows, kernel_cols, threads=grid)
