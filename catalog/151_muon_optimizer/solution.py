import torch


def solve(G, Buf, output, M, N, momentum):
    update = torch.add(G, Buf, alpha=momentum)
    normalized = update / (torch.linalg.vector_norm(update) + 1e-7)
    # Keep the smaller dimension on the left. The polynomial then squares
    # a min(M,N)-by-min(M,N) Gram matrix on each Newton-Schulz iteration.
    y = normalized.T if M > N else normalized
    for _ in range(5):
        gram = y @ y.T
        polynomial = torch.addmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
        y = torch.addmm(y, polynomial, y, beta=3.4445)
    output.copy_((y.T if M > N else y) * max(M, N) ** 0.5)
