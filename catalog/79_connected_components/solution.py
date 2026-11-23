import torch
from catalog.metal_shader import kernel


def solve(src, dst, labels, N, M):
    # Start with singleton trees. Each edge joins two roots; a second dispatch
    # looks up the minimum vertex ID reached from each original vertex.
    torch.arange(N, device=labels.device, dtype=labels.dtype, out=labels)
    if M:
        kernel(__file__, "unite")(src, dst, labels, threads=M)
        kernel(__file__, "compress")(labels, threads=N)
