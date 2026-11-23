import torch
import torch.nn.functional as F


def solve(input, output, N):
    # Both halves are views of the same storage. Request the Gaussian-CDF
    # activation (the benchmark uses erf), then multiply into the output.
    value, gate = input.chunk(2)
    torch.mul(value, F.gelu(gate, approximate="none"), out=output)
