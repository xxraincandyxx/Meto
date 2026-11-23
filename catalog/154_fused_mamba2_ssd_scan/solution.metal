#include <metal_stdlib>
using namespace metal;

kernel void state_scan(device const float *x, device const float *dt, device const float *bias,
                       device const float *a, device const float *b, device const float *c,
                       device const float *skip, device const float *z, device float *out,
                       constant uint &length, constant uint &heads, constant uint &width,
                       constant uint &groups, constant uint &state_dim,
                       uint job [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint channel = job % width, head = (job / width) % heads, batch = job / (width * heads);
    uint group = head / (heads / groups);
    float state[4] = {};
    for (uint t = 0; t < length; ++t) {
        uint token = batch * length + t, index = (token * heads + head) * width + channel;
        uint base = (token * groups + group) * state_dim;
        float raw = dt[token * heads + head] + bias[head];
        // Stable softplus gives the positive step size. exp(step*A) decays the
        // previous state while B injects the current input into each coordinate.
        float small = exp(-abs(raw));
        float step = max(raw, 0.0f) + (small < 1e-4f ? small : log(1 + small));
        float decay = exp(step * a[head]), input = x[index], contribution = 0;
        for (uint i = lane; i < state_dim; i += 32) {
            state[i / 32] = decay * state[i / 32] + (step * input) * b[base + i];
            contribution += state[i / 32] * c[base + i];
        }
        float response = simd_sum(contribution);
        if (lane == 0) {
            float gate = z[index];
            out[index] = (response + skip[head] * input) * gate / (1 + exp(-gate));
        }
    }
}
