import torch.nn.functional as F


def solve(images, patch_weight, patch_bias, cls_token, pos_embed, output,
          B, C, H, W, P, D):
    # A patch projection is a convolution whose kernel and stride both equal
    # the patch size. MPS can tile this directly from the original NCHW image,
    # avoiding a separate array of flattened patch vectors.
    patches = F.conv2d(images, patch_weight, patch_bias, stride=P)
    output[:, 0].copy_(cls_token + pos_embed[0])
    output[:, 1:].copy_(patches.flatten(2).transpose(1, 2) + pos_embed[1:])
