#include <metal_stdlib>
using namespace metal;

kernel void normalize(device const float *x, device const float *residual,
                      device const float *scale, device const float *bias, device float *out,
                      device float *combined, device float *mean, device float *rstd,
                      constant uint &cols, constant float &eps,
                      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint base = row * cols;
    float anchor = x[base] + residual[base];
    float partial = 0;
    for (uint c = lane; c < cols; c += 32) {
        float h = x[base + c] + residual[base + c];
        combined[base + c] = h;
        partial += h - anchor;
    }
    // Center around an actual sample before accumulating variance. Subtracting
    // two nearly equal squared moments would lose significant digits on rows
    // containing a large common offset.
    float shift = simd_sum(partial) / cols;
    float squares = 0;
    for (uint c = lane; c < cols; c += 32) {
        float centered = (combined[base + c] - anchor) - shift;
        squares += centered * centered;
    }
    float inverse = rsqrt(simd_sum(squares) / cols + eps);
    if (lane == 0) { mean[row] = anchor + shift; rstd[row] = inverse; }
    for (uint c = lane; c < cols; c += 32) {
        float normalized = ((combined[base + c] - anchor) - shift) * inverse;
        out[base + c] = normalized * scale[c] + bias[c];
    }
}
