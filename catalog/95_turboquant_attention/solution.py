import math
import torch


def solve(Q, K_idx, qjl_signs, gamma, Pi, S_mat, codebook, scores, B, S, D, C):
    # Move the two D-by-D transformations onto the queries. A decoding batch
    # typically has many more cached keys than queries, making these projections
    # cheaper while preserving the inner product via associativity.
    centroids = codebook[K_idx.long()]
    coarse = (Q @ Pi.T) @ centroids.T
    correction = (Q @ S_mat.T) @ qjl_signs.float().T
    torch.addcmul(coarse, correction, gamma, value=math.sqrt(math.pi/2) / D, out=scores)
