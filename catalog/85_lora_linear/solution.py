import torch


def solve(x, W, A, B, output, batch, d_in, d_out, rank, lora_scale):
    # Parenthesize the low-rank branch as (x Aᵀ) Bᵀ. Its intermediate width
    # is rank, which avoids constructing a dense weight update. addmm applies
    # the adapter scale inside the final matrix-multiply-and-add operation.
    base = x @ W.T
    narrow = x @ A.T
    torch.addmm(base, narrow, B.T, alpha=lora_scale, out=output)
