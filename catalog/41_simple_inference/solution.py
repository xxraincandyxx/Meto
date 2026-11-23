"""Run a supplied linear inference layer directly into its output buffer."""

import torch


def solve(input, model, output):
    # A Linear layer stores [output, input] weights. The transpose is a view
    # and the Metal matrix multiply accepts it directly. addmm includes the
    # bias by broadcasting it across the batch. Both branches reuse output.
    with torch.inference_mode():
        if model.bias is None:
            torch.mm(input, model.weight.T, out=output)
        else:
            torch.addmm(model.bias, input, model.weight.T, out=output)
