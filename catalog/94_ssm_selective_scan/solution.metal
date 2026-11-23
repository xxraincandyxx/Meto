#include <metal_stdlib>
using namespace metal;

kernel void solve(device const float* u, device const float* delta, device const float* a,
                  device const float* b, device const float* c, device const float* skip,
                  device float* y, constant uint& length, constant uint& channels,
                  constant uint& states, uint i [[thread_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]]) {
    uint pair = i / 32, batch = pair / channels, channel = pair % channels;
    float h[2] = {0, 0};
    for (uint t = 0; t < length; ++t) {
        uint token = batch * length + t, element = token * channels + channel;
        float value = u[element], dt = delta[element], mix = 0;
        for (uint slot = 0; slot < 2; ++slot) {
            uint s = lane + 32 * slot;
            if (s < states) {
                h[slot] = fma(exp(dt * a[channel * states + s]), h[slot],
                              dt * b[token * states + s] * value);
                mix = fma(c[token * states + s], h[slot], mix);
            }
        }
        // Sum across state coordinates; only lane zero commits the output.
        float sum = simd_sum(mix);
        if (lane == 0) y[element] = sum + skip[channel] * value;
    }
}
