import torch


def solve(x, gamma, W, w_scale, output, M, N, K):
    # RMS is a positive scalar per row. It cancels when the row is divided
    # by its own peak for quantization. Apply RMS once to the final row scale.
    weighted = x * gamma
    peak = weighted.abs().amax(dim=1, keepdim=True)
    scale = peak / 127
    safe_scale = torch.where(peak == 0, 1, scale)
    quantized = (weighted / safe_scale).round_().clamp_(-127, 127)
    inv_rms = (x.square().mean(1, keepdim=True) + 1e-6).rsqrt_()
    # Integer products up to 127*128 are exactly representable in FP32.
    # Use the Metal FP32 GEMM path and restore the scales after accumulation.
    torch.mm(quantized, W.float().T, out=output)
    output.mul_(scale * inv_rms).mul_(w_scale)
