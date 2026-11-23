#include <metal_stdlib>
using namespace metal;

kernel void activate(device const float* x, device float* y, uint i [[thread_position_in_grid]]) {
    // Both candidates come from the same input register. select lets each
    // lane choose the appropriate slope without materializing a boolean mask.
    float v = x[i];
    y[i] = select(v * 0.01f, v, v > 0.0f);
}
