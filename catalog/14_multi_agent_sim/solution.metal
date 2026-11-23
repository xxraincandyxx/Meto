#include <metal_stdlib>
using namespace metal;

kernel void advance(device const float4* agents, device float4* next, constant uint& n,
                    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    // Tiles turn 128 independent scans of global memory into one cooperative
    // load per group. Tail threads still participate in both barriers.
    threadgroup float4 tile[128];
    float4 self = i < n ? agents[i] : float4(0);
    float2 sum = 0; uint count = 0;
    for (uint base = 0; base < n; base += 128) {
        if (base + lane < n) tile[lane] = agents[base + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < min(128u, n - base); ++j) {
            float2 delta = self.xy - tile[j].xy;
            if (base + j != i && dot(delta, delta) < 25.0f) {
                sum += tile[j].zw;
                ++count;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float2 velocity = count ? mix(self.zw, sum / float(count), 0.05f) : self.zw;
    if (i < n) next[i] = float4(self.xy + velocity, velocity);
}
