"""Compute valid one-dimensional cross-correlation."""

import torch


def solve(input, kernel, output, input_size, kernel_size):
    # A long filter makes direct evaluation quadratic in the window width.
    # FFT convolution turns this into three transforms and an elementwise product.
    # Reversing the filter converts mathematical convolution to cross-correlation.
    if kernel_size >= 128 and output.numel() >= 4096:
        length = 1 << (input_size + kernel_size - 2).bit_length()
        signal = torch.fft.rfft(input, n=length)
        taps = torch.fft.rfft(kernel.flip((0,)), n=length)
        padded = torch.fft.irfft(signal * taps, n=length)
        output.copy_(padded[kernel_size - 1:input_size])
    else:
        # Batch and channel dimensions are both one. The view is free.
        values = torch.nn.functional.conv1d(input.view(1, 1, -1), kernel.view(1, 1, -1))
        output.copy_(values.flatten())
