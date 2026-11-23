#include <metal_stdlib>
using namespace metal;

kernel void rectify(device const float* x, device float* y, uint i [[thread_position_in_grid]]) {
    // The hardware maximum instruction performs the nonlinear activation in one
    // pass. Consecutive lanes read and write consecutive float32 elements.
    y[i] = max(0.0f, x[i]);
}
