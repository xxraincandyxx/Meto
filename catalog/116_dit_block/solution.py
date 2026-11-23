import torch.nn.functional as F


def solve(x, c, output, weights, batch_size, seq_len):
    width = x.size(-1)
    # The flat buffer contains consecutive (matrix, bias) pairs. These are
    # views; decoding a layer's parameters does not duplicate model weights.
    remaining = weights

    def linear(tensor, out_dim):
        nonlocal remaining
        in_dim = tensor.size(-1)
        count = out_dim * in_dim
        matrix = remaining[:count].view(out_dim, in_dim)
        bias = remaining[count:count + out_dim]
        remaining = remaining[count + out_dim:]
        return F.linear(tensor, matrix, bias)

    params = linear(F.silu(c), 6 * width).view(batch_size, 1, 6, width)
    shift1, stretch1, gate1, shift2, stretch2, gate2 = params.unbind(2)

    def condition(tensor, shift, stretch):
        norm = F.layer_norm(tensor, (width,), eps=1e-6)
        return norm.mul_(stretch + 1).add_(shift)

    packed = linear(condition(x, shift1, stretch1), 3 * width)
    heads = [part.reshape(batch_size, seq_len, 8, width // 8).transpose(1, 2)
             for part in packed.chunk(3, dim=-1)]
    context = F.scaled_dot_product_attention(*heads).transpose(1, 2).reshape_as(x)
    # Gates scale the contribution to each residual branch for this condition.
    residual = x + gate1 * linear(context, width)
    hidden = linear(condition(residual, shift2, stretch2), 4 * width)
    output.copy_(residual + gate2 * linear(F.gelu(hidden, approximate="tanh"), width))
