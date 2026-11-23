import torch


def solve(Q, K, V, dO, dQ, dK, dV, M, N, d):
    inv_scale = d ** -0.5
    probabilities = (Q @ K.T).mul_(inv_scale).softmax(dim=1)
    torch.mm(probabilities.T, dO, out=dV)
    weighted_grad = (dO @ V.T).mul_(probabilities)
    # Subtract the projection along the probability vector. This is the
    # softmax Jacobian applied to a vector, using one row sum instead of an
    # N-by-N Jacobian per token.
    weighted_grad.addcmul_(probabilities, weighted_grad.sum(1, keepdim=True), value=-1)
    weighted_grad.mul_(inv_scale)
    torch.mm(weighted_grad, K, out=dQ)
    torch.mm(weighted_grad.T, Q, out=dK)
