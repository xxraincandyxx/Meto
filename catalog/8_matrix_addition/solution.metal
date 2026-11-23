#include <metal_stdlib>
using namespace metal;

kernel void add(device const float* a, device const float* b, device float* c,
                constant uint& count, uint packet [[thread_position_in_grid]]) {
    uint i = 4 * packet;
    if (i + 3 < count) {
        // Matrix rows are contiguous: a packet may even cross a row boundary.
        // Buffer starts and all packet offsets are aligned to sixteen bytes.
        reinterpret_cast<device float4*>(c)[packet] =
            reinterpret_cast<device const float4*>(a)[packet] +
            reinterpret_cast<device const float4*>(b)[packet];
    } else {
        // A square with an odd side can leave a short final packet.
        for (; i < count; ++i) c[i] = a[i] + b[i];
    }
}
