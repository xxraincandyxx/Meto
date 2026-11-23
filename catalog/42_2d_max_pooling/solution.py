"""Take one maximum per pooling window across a batch of images."""

import torch


def solve(input, output, N, C, H, W, kernel_size, stride, padding):
    # NCHW views describe batches and channels without copying the image. MPS
    # implements the sliding-window maximum; its implicit padding is negative
    # infinity so negative border pixels still compete correctly for the maximum.
    pooled = torch.nn.functional.max_pool2d(input.view(N, C, H, W), kernel_size, stride, padding)
    output.copy_(pooled.flatten())
