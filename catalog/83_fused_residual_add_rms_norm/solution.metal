#include <metal_stdlib>
using namespace metal;

kernel void solve(device const float* x, device const float* residual,
                  device const float* weight, device float* out,
                  constant uint& columns, constant float& eps,
                  uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint base = row * columns;
    float squares = 0;
    for (uint c = lane; c < columns; c += 32) {
        float z = x[base + c] + residual[base + c];
        squares = fma(z, z, squares);
    }
    float inverseRMS = rsqrt(simd_sum(squares) / float(columns) + eps);
    // Reconstruct the cheap addition as we write the output; a temporary
    // N-by-C residual array would need both a device write and a device read.
    for (uint c = lane; c < columns; c += 32)
        out[base + c] = (x[base + c] + residual[base + c]) * inverseRMS * weight[c];
}
