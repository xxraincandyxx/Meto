#include <metal_stdlib>
using namespace metal;

// A binary tree halves the number of partial sums on each round. All threads
// reach each barrier, including padding threads whose input contribution is zero.
kernel void reduceSum(device const float *a, device float *partial, constant uint &n,
                      uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                      uint g [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    sums[t] = i < n ? a[i] : 0;
    for (uint stride = 128; stride > 0; stride /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < stride) sums[t] += sums[t+stride];
    }
    if (t == 0) partial[g] = sums[0];
}
