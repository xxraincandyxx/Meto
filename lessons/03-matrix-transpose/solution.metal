#include <metal_stdlib>
using namespace metal;

kernel void transpose(device const float *a, device float *out,
                      constant uint &rows, constant uint &cols,
                      uint2 pos [[thread_position_in_grid]]) {
    // Each thread transposes a 4x4 rectangle in registers. Four short contiguous
    // reads become four short contiguous writes. Packed vectors are aligned
    // like scalars, which matters when an odd stride shifts the next row by 4 bytes.
    uint x = pos.x * 4, y = pos.y * 4;
    if (x >= cols || y >= rows) return;
    if (x + 4 <= cols && y + 4 <= rows) {
        float4 lanes[4];
        for (uint i = 0; i < 4; ++i)
            lanes[i] = float4(*reinterpret_cast<device const packed_float4 *>(a + (y+i)*cols + x));
        for (uint i = 0; i < 4; ++i)
            *reinterpret_cast<device packed_float4 *>(out + (x+i)*rows + y) =
                packed_float4(lanes[0][i], lanes[1][i], lanes[2][i], lanes[3][i]);
    } else {
        // Only the bottom and right edges take this path. A full vector there
        // could cross into a neighboring row or beyond the buffer.
        for (uint row = y; row < min(y+4, rows); ++row)
            for (uint col = x; col < min(x+4, cols); ++col) out[col*rows+row] = a[row*cols+col];
    }
}
