#include <metal_stdlib>
using namespace metal;

kernel void solve(device const packed_float3* position, device const float* mass,
                  device packed_float3* force, constant uint& count,
                  uint i [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float4 particles[128];
    float3 here = i < count ? float3(position[i]) : float3(0), sum = 0;
    for (uint base = 0; base < count; base += 128) {
        uint j = base + lane;
        if (j < count) particles[lane] = float4(float3(position[j]), mass[j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Load a particle once per group, then reuse it for up to 128 targets.
        // The displacement vanishes for self-interaction. Softening keeps the
        // reciprocal distance finite for coincident particles as well.
        if (i < count) {
            for (uint k = 0; k < min(128u, count - base); ++k) {
                float3 delta = particles[k].xyz - here;
                float inv = rsqrt(dot(delta, delta) + 1e-6f);
                sum += delta * (particles[k].w * inv * inv * inv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < count) force[i] = packed_float3(sum);
}
