import torch
import torch.nn.functional as F


def solve(X, W, targets, loss, N, D, V):
    # Bound scratch space for logits to about four million floats. Grouping
    # rows still gives Metal's GEMM a substantial tile to process at once.
    tile = max(1, min(N, (1 << 22) // V))
    total = torch.zeros((), device=X.device)
    for start in range(0, N, tile):
        logits = F.linear(X[start:start + tile], W)
        target = targets[start:start + tile].long()
        total.add_(F.cross_entropy(logits, target, reduction="sum"))
    loss.copy_(total.div_(N))
