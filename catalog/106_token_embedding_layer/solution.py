from torch.nn.functional import embedding, layer_norm


def solve(token_ids, position_ids, token_embeddings, position_embeddings, gamma,
          beta, output, B, T, V, P, D, eps):
    # Look up position vectors once. Broadcasting their leading dimension
    # shares these T rows across all B sequences without repeating the table.
    words = embedding(token_ids, token_embeddings)
    words.add_(embedding(position_ids, position_embeddings))
    output.copy_(layer_norm(words, (D,), gamma, beta, eps))
