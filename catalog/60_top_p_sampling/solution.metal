#include <metal_stdlib>
using namespace metal;

inline uint pcg(uint x) {
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

kernel void solve(
    device const float* probabilities,
    device const float* cumulative,
    device const long* indices,
    device const float* threshold,
    device const int* seed,
    device float* score,
    uint i [[thread_position_in_grid]]) {
    // The sorted prefix before this token determines whether the token is in
    // the nucleus. Comparing it with p includes the first token to cross p.
    if (i != 0 && cumulative[i - 1] >= threshold[0]) {
        score[i] = -INFINITY;
        return;
    }
    uint bits = pcg(pcg(as_type<uint>(seed[0])) + uint(indices[i]));
    // Evaluate -log(1 - distance) by its first two Taylor terms near 1.
    // Counting the distance to UINT_MAX
    // prevents a near-one sample from rounding to exactly one in float32.
    float distance = (float(0xffffffffu - bits) + 0.5f) * 0x1p-32f;
    float clock = distance < 0.0001f
        ? distance * (1.0f + 0.5f * distance)
        : -log((float(bits) + 0.5f) * 0x1p-32f);
    score[i] = probabilities[i] / clock;
}
