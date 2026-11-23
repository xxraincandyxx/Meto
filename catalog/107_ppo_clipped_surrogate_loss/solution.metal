#include <metal_stdlib>
using namespace metal;

kernel void ppo(device const float *advantage, device const float *policy, device const float *old,
                 device float *parts, constant uint &count, constant float &clip,
                 uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // 256 threads cover four contiguous stripes. Neighboring lanes visit
    // consecutive tokens, while the bounds check handles a short last stripe.
    float sum = 0;
    for (uint i = group * 1024 + tid; i < min(count, (group + 1) * 1024); i += 256) {
        float a = advantage[i], ratio = exp(policy[i] - old[i]);
        float clipped = a >= 0 ? min(ratio, 1 + clip) : max(ratio, 1 - clip);
        sum -= a * clipped;
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
