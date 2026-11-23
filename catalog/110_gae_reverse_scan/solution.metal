#include <metal_stdlib>
using namespace metal;

kernel void local_scan(device const float *reward, device const float *value,
                       device float *out, device float *totals, constant uint &s,
                       constant uint &chunks, constant float &gamma, constant float &decay,
                       uint job [[thread_position_in_grid]]) {
    uint row = (job / chunks) * s, begin = (job % chunks) * 64;
    float estimate = 0;
    for (int t = int(min(s, begin + 64)) - 1; t >= int(begin); --t) {
        float next = uint(t + 1) < s ? value[row + t + 1] : 0;
        float delta = reward[row + t] + gamma * next - value[row + t];
        estimate = delta + decay * estimate;
        out[row + t] = estimate;
    }
    totals[job] = estimate;
}

kernel void carry_scan(device float *totals, constant uint &chunks,
                       constant float &chunk_decay, uint row [[thread_position_in_grid]]) {
    // Replace each chunk's local total with the advantage at the start of the
    // following chunk. Only the final chunk may be short; its outgoing carry is zero.
    float after = 0;
    for (int k = int(chunks) - 1; k >= 0; --k) {
        uint index = row * chunks + k;
        float current = totals[index] + chunk_decay * after;
        totals[index] = after;
        after = current;
    }
}

kernel void apply_carry(device float *out, device const float *totals, constant uint &s,
                        constant uint &chunks, constant float &decay,
                        uint idx [[thread_position_in_grid]]) {
    uint t = idx % s, chunk = t / 64;
    float correction = totals[(idx / s) * chunks + chunk];
    out[idx] += correction * powr(decay, float(min(s, 64 * (chunk + 1)) - t));
}
