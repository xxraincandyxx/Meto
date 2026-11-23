import torch
from catalog.metal_shader import kernel


def solve(x, W_gate, W_up, W_down, output, M, d_model, d_ffn):
    # Each matrix product uses the tiled GEMM backend. The Metal gate kernel
    # reads both projected activations and overwrites one, saving a hidden
    # activation buffer and a separate multiply dispatch.
    gate = x @ W_gate
    up = x @ W_up
    kernel(__file__)(gate, up, threads=M * d_ffn)
    torch.mm(gate, W_down, out=output)
