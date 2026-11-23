import torch


def solve(input, output, N, window_size):
    # Prefix[j] contains the sum before index j. Subtract two shifted views
    # to compute every fixed-width window at once, including the final window.
    prefix = torch.nn.functional.pad(torch.cumsum(input, 0, dtype=torch.int64), (1, 0))
    output.copy_((prefix[window_size:] - prefix[:-window_size]).amax())
