#include <metal_stdlib>
using namespace metal;

kernel void adamw(device float *params, device const float *grad, device float *m,
                  device float *v, constant float &lr, constant float &b1,
                  constant float &b2, constant float &eps, constant float &shrink,
                  constant float &correct1, constant float &correct2,
                  uint i [[thread_position_in_grid]]) {
    // Cache each gradient in a register. Updates for different parameters are
    // independent, so a single dispatch can persist all three updated arrays.
    float g = grad[i];
    float first = b1 * m[i] + (1 - b1) * g;
    float second = b2 * v[i] + (1 - b2) * g * g;
    m[i] = first;
    v[i] = second;
    params[i] = params[i] * shrink - lr * (first * correct1) / (sqrt(second * correct2) + eps);
}
