import torch
import torch.nn.functional as F


def solve(Q, K, V, output, M, d):
    # Move the parentheses in (Q Kᵀ) V. For long sequences this creates
    # a d-by-d summary instead of an M-by-M attention matrix. One extra
    # value channel carries the normalizer through the same pair of GEMMs.
    # The negative branch is exp(x). Computing it directly preserves small
    # positive features even when ELU(x) itself would round to exactly -1.
    queries = torch.where(Q < 0, Q.exp(), Q + 1)
    keys = torch.where(K < 0, K.exp(), K + 1)
    values = F.pad(V, (0, 1), value=1)
    weighted = queries @ (keys.T @ values)
    torch.div(weighted[:, :d], weighted[:, d:], out=output)
