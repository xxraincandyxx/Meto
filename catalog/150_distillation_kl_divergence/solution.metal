#include <metal_stdlib>
using namespace metal;

kernel void divergence(device const float *teacher, device const float *student,
                       device float *out, constant uint &vocab, constant float &inverse_temp,
                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // Subtract each distribution's own maximum so all exponentials lie in [0,1].
    size_t base = size_t(row) * vocab;
    float p_max = -INFINITY, q_max = -INFINITY;
    for (uint j = tid; j < vocab; j += 256) {
        p_max = max(p_max, teacher[base + j]);
        q_max = max(q_max, student[base + j]);
    }
    threadgroup float2 maxima[8];
    p_max = simd_max(p_max);
    q_max = simd_max(q_max);
    if (lane == 0) maxima[sid] = float2(p_max, q_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 candidate = lane < 8 ? maxima[lane] : float2(-INFINITY);
    p_max = simd_max(candidate.x);
    q_max = simd_max(candidate.y);

    // For shifted logits p and q the divergence is
    // sum(exp(p)*(p-q))/Zp + log(Zq) - log(Zp).
    // Accumulate its three sufficient statistics together in registers.
    float p_total = 0, q_total = 0, gap = 0;
    for (uint j = tid; j < vocab; j += 256) {
        float p = (teacher[base + j] - p_max) * inverse_temp;
        float q = (student[base + j] - q_max) * inverse_temp;
        float mass = exp(p);
        p_total += mass;
        q_total += exp(q);
        gap += mass * (p - q);
    }
    threadgroup float3 partials[8];
    float3 part = float3(simd_sum(p_total), simd_sum(q_total), simd_sum(gap));
    if (lane == 0) partials[sid] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        float3 v = lane < 8 ? partials[lane] : float3(0);
        float zp = simd_sum(v.x), zq = simd_sum(v.y), moment = simd_sum(v.z);
        if (lane == 0) out[row] = moment / zp + log(zq / zp);
    }
}
