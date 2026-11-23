#include <metal_stdlib>
using namespace metal;

kernel void assign_points(device const float* px, device const float* py,
                          device const float* cx, device const float* cy,
                          device int* labels, constant uint& k,
                          uint point [[thread_position_in_grid]]) {
    float2 p(px[point], py[point]);
    float best = INFINITY; int winner = 0;
    // All lanes read the same centroid in an iteration. Keep only the best
    // distance in a register, so scratch space never grows as points * k.
    for (uint c = 0; c < k; ++c) {
        float2 offset = p - float2(cx[c], cy[c]);
        float distance = dot(offset, offset);
        if (distance < best) { best = distance; winner = int(c); }
    }
    labels[point] = winner;
}
