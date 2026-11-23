from torch.nn.functional import group_norm


def solve(X, gamma, beta, Y, N, C, H, W, G, eps):
    # NCHW already places all spatial samples of adjacent channels together.
    # The group-normalization primitive reduces C/G * H * W values per group
    # and broadcasts the channel-specific affine parameters across its pixels.
    Y.copy_(group_norm(X, num_groups=G, weight=gamma, bias=beta, eps=eps))
