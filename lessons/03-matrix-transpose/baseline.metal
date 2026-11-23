#include <metal_stdlib>
using namespace metal;

// Swap the row and column coordinates. Adjacent input columns become different
// output rows, so neighboring threads write far apart for a tall matrix.
kernel void transpose(device const float *a, device float *out, constant uint &rows,
                      constant uint &cols, uint2 p [[thread_position_in_grid]]) {
    if (p.x < cols && p.y < rows) out[p.x*rows + p.y] = a[p.y*cols + p.x];
}
