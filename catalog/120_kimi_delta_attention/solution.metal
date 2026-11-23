#include <metal_stdlib>
using namespace metal;

kernel void delta_rule(device const float *q, device const float *k, device const float *v,
                       device const float *alpha, device const float *beta, device float *out,
                       constant uint &length, constant uint &d, uint job [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint batch = job / d, col = job % d;
    // A column has at most 128 entries: keep four entries per SIMD lane in
    // registers across the entire sequence. Nothing writes a dense state to DRAM.
    float state[4] = {};
    for (uint t = 0; t < length; ++t) {
        uint token = batch * length + t, base = token * d;
        float prediction = 0;
        for (uint i = lane; i < d; i += 32) {
            state[i / 32] *= alpha[base + i];
            prediction += state[i / 32] * k[base + i];
        }
        float error = beta[token] * (v[base + col] - simd_sum(prediction));
        float projected = 0;
        for (uint i = lane; i < d; i += 32) {
            state[i / 32] += k[base + i] * error;
            projected += q[base + i] * state[i / 32];
        }
        float result = simd_sum(projected);
        if (lane == 0) out[base + col] = result;
    }
}
