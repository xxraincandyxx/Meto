import torch


def solve(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C):
    # Reuse the statistics saved during the forward pass. The affine
    # parameters are shared by all rows, so their gradients reduce vertically.
    normalized = (x - mean[:, None]) * rstd[:, None]
    torch.sum(dy, dim=0, out=dbeta)
    torch.sum(dy * normalized, dim=0, out=dgamma)
    torch.mul(dy, gamma, out=dx)
    center = dx.mean(dim=1, keepdim=True)
    projection = (dx * normalized).mean(dim=1, keepdim=True)
    dx.sub_(center).addcmul_(normalized, projection, value=-1).mul_(rstd[:, None])
