#include <metal_stdlib>
using namespace metal;

kernel void score(device const float* q, device const char* keys, device const float* scales,
                  device float* scores, constant uint& length, constant uint& dim,
                  uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint token = i / 32, head = token / length;
    float dotProduct = 0;
    for (uint d = lane; d < dim; d += 32)
        dotProduct = fma(q[head * dim + d], float(keys[token * dim + d]), dotProduct);
    float sum = simd_sum(dotProduct);
    if (lane == 0) scores[token] = sum * scales[token] * rsqrt(float(dim));
}

kernel void mix(device const float* weights, device const char* values,
                device float* out, constant uint& length, constant uint& dim,
                uint i [[thread_position_in_grid]]) {
    uint head = i / dim, feature = i % dim;
    float sum = 0;
    // Lanes read adjacent feature bytes. The weight is shared by every
    // output feature of the same head, so the cache can broadcast it.
    for (uint t = 0; t < length; ++t) {
        uint token = head * length + t;
        sum = fma(weights[token], float(values[token * dim + feature]), sum);
    }
    out[i] = sum;
}
