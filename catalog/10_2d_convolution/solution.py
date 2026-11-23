"""Slide a two-dimensional filter over all fully covered image windows."""

import torch


def solve(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols):
    # MPS convolution takes [batch, channels, height, width]. Describing our
    # single image and filter with those strides lets its optimized spatial kernel
    # reuse overlapping neighborhoods without allocating an unfolded patch matrix.
    image = input.view(1, 1, input_rows, input_cols)
    taps = kernel.view(1, 1, kernel_rows, kernel_cols)
    output.copy_(torch.nn.functional.conv2d(image, taps).flatten())
