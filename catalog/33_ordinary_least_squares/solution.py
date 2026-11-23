"""Fit least-squares coefficients using a compact system of normal equations."""

import torch


def solve(X, y, beta, n_samples, n_features):
    # Compress many sample rows into the small feature-by-feature Gram matrix.
    # Only the two matrix products touch all samples. The solve then works on
    # n_features unknowns; it avoids ever materializing an inverse matrix.
    design = X.view(n_samples, n_features)
    gram = design.T @ design
    rhs = design.T @ y.reshape(-1)
    solution, _ = torch.linalg.solve_ex(gram, rhs)
    beta.copy_(solution)
