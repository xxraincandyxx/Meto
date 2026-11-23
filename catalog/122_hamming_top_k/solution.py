import torch
from catalog.metal_shader import kernel


def solve(codes, query, output, N, k):
    distances = torch.empty(N, dtype=torch.int32, device=codes.device)
    kernel(__file__, "distance")(codes, query, distances, threads=N)
    # Stable sorting breaks equal-distance ties by original row index.
    output.copy_(distances.argsort(stable=True)[:k])
