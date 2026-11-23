#include <metal_stdlib>
using namespace metal;

kernel void nearest(device const packed_float3* points, device int* indices, constant uint& n,
                    uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    // packed_float3 follows the twelve-byte XYZ stride of the input tensor.
    // Group-local staging lets 128 queries reuse each batch of candidate points.
    threadgroup float3 tile[128];
    float3 query = i < n ? float3(points[i]) : float3(0);
    float best = INFINITY;
    uint answer = 0;  // The one-point case has no other point.
    for (uint base = 0; base < n; base += 128) {
        if (base + t < n) tile[t] = float3(points[base + t]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < min(n - base, 128u); ++j) {
            float3 delta = query - tile[j];
            float distance = dot(delta, delta);
            if (base + j != i && distance < best) {
                best = distance;
                answer = base + j;  // Strict < keeps the first index on ties.
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < n) indices[i] = int(answer);
}
