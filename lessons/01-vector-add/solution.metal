#include <metal_stdlib>
using namespace metal;

// Addition moves twelve bytes for one arithmetic operation (two float reads
// and one write). A thread processes four adjacent values to amortize indexing.
// Buffer starts are aligned by Metal; the packed path begins at a multiple of 4.
kernel void add(device const float *a, device const float *b, device float *out,
                constant uint &n, uint i [[thread_position_in_grid]]) {
    if (i < n / 4) {
        device const float4 *av = reinterpret_cast<device const float4 *>(a);
        device const float4 *bv = reinterpret_cast<device const float4 *>(b);
        reinterpret_cast<device float4 *>(out)[i] = av[i] + bv[i];
    } else if (i == n / 4) {
        // Only one thread owns the incomplete vector. A float4 load here would
        // read beyond the allocation for lengths such as 3 or 1027.
        for (uint j = 4 * i; j < n; ++j) out[j] = a[j] + b[j];
    }
}
