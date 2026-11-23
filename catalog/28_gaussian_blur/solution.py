"""Blur a single-channel image by applying its supplied two-dimensional filter."""

import torch


def solve(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols):
    # Half a window of implicit zero padding keeps the output centered and the
    # same size. The caller supplies all weights: this also handles kernels that
    # cannot be factored into horizontal and vertical one-dimensional filters.
    image = input.view(1, 1, input_rows, input_cols)
    weights = kernel.view(1, 1, kernel_rows, kernel_cols)
    blurred = torch.nn.functional.conv2d(image, weights, padding=(kernel_rows // 2, kernel_cols // 2))
    output.copy_(blurred.flatten())
