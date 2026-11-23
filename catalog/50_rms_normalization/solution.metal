#include <metal_stdlib>
using namespace metal;

kernel void normalize(device const float* x, device float* result, constant float& sum,
                      constant float& gain, constant float& bias, constant float& eps,
                      constant uint& n, uint i [[thread_position_in_grid]]) {
    // All lanes read the same small set of constants. Fuse reciprocal root,
    // gain and bias into the pass that actually writes the normalized vector.
    float inv_rms = rsqrt(sum / float(n) + eps);
    result[i] = fma(x[i], gain * inv_rms, bias);
}
