import torch


def solve(input, output, H, W):
    # Integrate in each dimension. The first scan works on contiguous rows;
    # the second accumulates those row prefixes vertically to cover rectangles.
    torch.cumsum(input.cumsum(dim=1), dim=0, out=output)
