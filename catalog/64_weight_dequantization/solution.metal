#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x,
    device const float* scales,
    device float* y,
    constant uint& columns,
    constant uint& tile,
    uint i [[thread_position_in_grid]]) {
    // Map each input coordinate to its scale tile on demand. The ceiling
    // division accounts for the partial tile at the right edge.
    uint row = i / columns, column = i % columns;
    uint scaleColumns = (columns + tile - 1) / tile;
    y[i] = x[i] * scales[(row / tile) * scaleColumns + column / tile];
}
