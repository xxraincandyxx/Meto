import torch
import torch.nn.functional as F


def solve(Q, K, V, output, M, d):
    # Move the parentheses in (Q Kᵀ) V. For long sequences this creates
    # a d-by-d summary instead of an M-by-M attention matrix. One extra
    # value channel carries the normalizer through the same pair of GEMMs.
    queries = F.elu(Q) + 1
    keys = F.elu(K) + 1
    # The negative branch is exp(x). Evaluating it directly avoids subtracting
    # almost equal numbers when ELU(x) is very close to -1.
    queries = torch.where(Q < 0, Q.exp(), queries)
    keys = torch.where(K < 0, K.exp(), keys)
    values = F.pad(V, (0, 1), value=1)
    weighted = queries @ (keys.T @ values)
    torch.div(weighted[:, :d], weighted[:, d:], out=output)
