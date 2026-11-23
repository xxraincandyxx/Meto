#include <metal_stdlib>
using namespace metal;

// A row of A meets a column of B at one output element. In row-major storage,
// advancing the row index of B jumps forward by its row length n.
kernel void multiply(device const float *a, device const float *b, device float *c,
                     constant uint &m, constant uint &k, constant uint &n,
                     uint2 p [[thread_position_in_grid]]) {
    if (p.x >= n || p.y >= m) return;
    float sum = 0;
    for (uint t = 0; t < k; ++t) sum = fma(a[p.y*k+t], b[t*n+p.x], sum);
    c[p.y*n+p.x] = sum;
}
