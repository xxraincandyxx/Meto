"""Run a supplied inference module on its device-resident inputs."""

import torch


def solve(input, model, output):
    # The runner places both parameters and inputs on Metal. Evaluation requires
    # only a forward pass: skipping autograd avoids retaining intermediate data
    # for a backward pass. The module may contain a bias or use only weights.
    with torch.inference_mode():
        output.copy_(model(input))
