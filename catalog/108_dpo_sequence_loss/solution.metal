#include <metal_stdlib>
using namespace metal;

kernel void dpo(device const float *chosen, device const float *rejected,
                device const float *chosen_ref, device const float *rejected_ref,
                 device float *parts, constant uint &count, constant float &beta,
                 uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // 256 threads cover four contiguous stripes. Neighboring lanes visit
    // consecutive tokens, while the bounds check handles a short last stripe.
    float sum = 0;
    for (uint i = group * 1024 + tid; i < min(count, (group + 1) * 1024); i += 256) {
        float margin = beta * ((rejected[i] - chosen[i]) + (chosen_ref[i] - rejected_ref[i]));
        // softplus(margin) without overflow or losing a tiny positive tail.
        float tail = exp(-abs(margin));
        sum += max(margin, 0.0f) + (tail < 1e-4f ? tail : log(1 + tail));
    }
    threadgroup float simd_totals[8];
    float subtotal = simd_sum(sum);
    if (lane == 0) simd_totals[sid] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The first SIMD group folds the eight subtotals into one partial mean.
    // Summing these scaled partials completes the global mean on the GPU.
    if (sid == 0) {
        float block_sum = simd_sum(lane < 8 ? simd_totals[lane] : 0.0f);
        if (lane == 0) parts[group] = block_sum / count;
    }
}
