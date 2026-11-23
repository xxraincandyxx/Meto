import torch


def solve(x, weight, dy, dx, dweight, N, C, eps):
    inv_rms = (x.square().mean(1, keepdim=True) + eps).rsqrt_()
    normalized = x * inv_rms
    # Differentiate the row's normalization factor once per row. Removing
    # this radial component leaves the part of the gradient tangent to its
    # normalized direction. Each channel's weight gradient sums over rows.
    incoming = torch.mul(dy, weight, out=dx)
    radial = (incoming * normalized).mean(1, keepdim=True)
    dx.addcmul_(normalized, radial, value=-1).mul_(inv_rms)
    torch.sum(dy * normalized, dim=0, out=dweight)
