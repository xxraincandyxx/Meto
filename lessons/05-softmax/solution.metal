#include <metal_stdlib>
using namespace metal;

// Each tile is represented by (largest logit, sum measured relative to it).
// The same reduction handles both a tile of logits and the list of tile summaries.
template<bool useMax>
inline float combine(float v, threadgroup float *scratch, uint tid, uint lane, uint sg,
                     uint groupCount, uint width) {
    float sub = useMax ? simd_max(v) : simd_sum(v);
    if (!lane) scratch[sg] = sub;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!sg) {
        float part = useMax ? -INFINITY : 0.f;
        for (uint i = lane; i < groupCount; i += width)
            part = useMax ? max(part, scratch[i]) : part + scratch[i];
        sub = useMax ? simd_max(part) : simd_sum(part);
        if (!tid) scratch[0] = sub;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

kernel void summarize(device const float *x, device float2 *stats, constant uint &n,
                      uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint groups [[simdgroups_per_threadgroup]], uint width [[threads_per_simdgroup]]) {
    threadgroup float scratch[64];
    float values[4], high = -INFINITY;
    for (uint j = 0; j < 4; ++j) {
        uint i = block*1024 + j*256 + tid;
        values[j] = i < n ? x[i] : -INFINITY;
        high = max(high, values[j]);
    }
    high = combine<true>(high, scratch, tid, lane, sg, groups, width);
    float total = 0;
    for (uint j = 0; j < 4; ++j) total += exp(values[j] - high);
    total = combine<false>(total, scratch, tid, lane, sg, groups, width);
    if (!tid) stats[block] = float2(high, total);
}

kernel void mergeStats(device const float2 *stats, device float2 *global, constant uint &blocks,
                       uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]],
                       uint width [[threads_per_simdgroup]]) {
    threadgroup float scratch[64];
    float high = -INFINITY;
    for (uint i = tid; i < blocks; i += 256) high = max(high, stats[i].x);
    high = combine<true>(high, scratch, tid, lane, sg, groups, width);
    // A block measured at max m contributes sum * exp(m - globalMax).
    float total = 0;
    for (uint i = tid; i < blocks; i += 256) total += stats[i].y * exp(stats[i].x - high);
    total = combine<false>(total, scratch, tid, lane, sg, groups, width);
    if (!tid) global[0] = float2(high, 1.f / total);
}

kernel void normalize(device const float *x, device const float2 *global, device float *out,
                      constant uint &n, uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = exp(x[i] - global[0].x) * global[0].y;
}
