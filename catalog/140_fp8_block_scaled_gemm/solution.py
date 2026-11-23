import torch
from catalog.metal_shader import kernel


def solve(x_q, x_scales, w_q, w_scales, y, M, N, K):
    # Unpack each byte once before the large matrix product reuses it across
    # output columns/rows. FP32 intermediates preserve range until the final
    # FP16 output conversion, including when block scales are far apart.
    def dequantize(encoded, scales):
        result = torch.empty_like(encoded, dtype=torch.float32)
        kernel(__file__, "decode")(encoded, scales, result, threads=encoded.numel())
        return result

    x = dequantize(x_q, x_scales)
    weights = dequantize(w_q, w_scales)
    y.copy_(x @ weights.T)
