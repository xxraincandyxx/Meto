import torch.nn.functional as F


def solve(input, weight, bias, output, N, C, eps):
    # A contiguous row is one normalization region. Supplying the learned
    # scale and bias to layer_norm lets the backend apply them as the final
    # affine step with the row's mean and reciprocal standard deviation.
    output.copy_(F.layer_norm(input, (C,), weight, bias, eps))
