from catalog.metal_shader import kernel


def solve(x, W_gate_up, output, M, H, I):
    # Both projections share the same input. One wide GEMM makes a packed
    # matrix whose two halves become gate and value for a fused activation.
    projected = x @ W_gate_up
    kernel(__file__, "activate")(projected, output, I, threads=M * I)
