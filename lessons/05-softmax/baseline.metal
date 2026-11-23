#include <metal_stdlib>
using namespace metal;

// One group can synchronize the whole calculation. Threads visit the input in
// stripes, reduce its maximum, then repeat for the sum of shifted exponentials.
kernel void softmax(device const float *x, device float *out, constant uint &n,
                    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[256];
    float high = -INFINITY;
    for (uint i = lane; i < n; i += 256) high = max(high, x[i]);
    scratch[lane] = high;
    for (uint step = 128; step; step /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < step) scratch[lane] = max(scratch[lane], scratch[lane+step]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    high = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = lane; i < n; i += 256) total += exp(x[i] - high);
    scratch[lane] = total;
    for (uint step = 128; step; step /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < step) scratch[lane] += scratch[lane+step];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < n; i += 256) out[i] = exp(x[i] - high) / scratch[0];
}
