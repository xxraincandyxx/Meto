#include <metal_stdlib>
using namespace metal;

kernel void overlaps(device const float4 *boxes, device uint *masks, constant uint &n,
                     constant uint &words, constant float &threshold,
                     uint job [[thread_position_in_grid]]) {
    uint row = job / words, first = (job % words) * 32;
    float4 a = boxes[row];
    float area = (a.z - a.x) * (a.w - a.y);
    uint bits = 0;
    // A word records which lower-priority boxes would be suppressed. Pack
    // 32 comparisons into one store, and omit pairs above the score diagonal.
    for (uint j = max(row + 1, first); j < min(n, first + 32); ++j) {
        float4 b = boxes[j];
        float2 span = max(min(a.zw, b.zw) - max(a.xy, b.xy), float2(0));
        float intersection = span.x * span.y;
        float total = area + (b.z - b.x) * (b.w - b.y) - intersection;
        if (intersection / total > threshold) bits |= 1u << (j - first);
    }
    masks[job] = bits;
}

kernel void select_boxes(device const int *order, device const uint *masks,
                         device uint *suppressed, device int *keep,
                         constant uint &n, constant uint &words,
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint take;
    for (uint row = 0; row < n; ++row) {
        if (tid == 0) {
            take = ((suppressed[row / 32] >> (row % 32)) & 1u) == 0;
            keep[order[row]] = int(take);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Greedy choices are sequential. The expensive operation after each
        // accepted choice (merging its mask) is parallel across the group.
        if (take) {
            for (uint w = row / 32 + tid; w < words; w += 256)
                suppressed[w] |= masks[row * words + w];
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
