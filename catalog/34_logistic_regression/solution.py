"""Fit a regularized binary classifier with weighted Newton steps."""

import torch


def solve(X, y, beta, n_samples, n_features):
    beta.zero_()
    ridge = 1e-6
    identity = torch.eye(n_features, device=X.device, dtype=X.dtype)
    # The curvature matrix is X^T diag(p*(1-p)) X plus a ridge term. Scale
    # the rows first to avoid allocating the much larger samples-by-samples
    # diagonal matrix. A small curvature floor keeps confident rows usable.
    for _ in range(1000):
        p = (X @ beta).sigmoid()
        gradient = torch.addmv(beta * ridge, X.T, p - y)
        weight = (p * (1 - p)).clamp_min_(1e-8)
        hessian = torch.addmm(identity, X.T, weight[:, None] * X, beta=ridge)
        delta, _ = torch.linalg.solve_ex(hessian, gradient)
        updated = beta - delta
        converged = torch.linalg.vector_norm(updated - beta) < 1e-8
        beta.copy_(updated)
        if converged.item():
            break
