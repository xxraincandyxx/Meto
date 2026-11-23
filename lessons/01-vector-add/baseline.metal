#include <metal_stdlib>
using namespace metal;

// The host binds buffers in the same order as these arguments. Each thread gets
// its own grid index i and writes one sum; threads never need to wait for a peer.
kernel void add(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                device float *out [[buffer(2)]], constant uint &n [[buffer(3)]],
                uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] + b[i];
}
