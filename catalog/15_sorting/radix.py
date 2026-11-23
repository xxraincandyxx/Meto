"""Sort a float vector in ascending order on the Metal device."""

from importlib import import_module

import torch

from catalog.metal_shader import kernel

sort_keys = import_module("catalog.36_radix_sort.solution").solve


def solve(data, N):
    # Convert float ordering to unsigned ordering for the radix passes in
    # lesson 36. Decoding in place restores the sorted floating-point words.
    keys = torch.empty(N, dtype=torch.uint32, device=data.device)
    kernel(__file__, "encode")(data, keys, threads=N)
    sort_keys(keys, data.view(torch.uint32), N)
    kernel(__file__, "decode")(data, threads=N)
