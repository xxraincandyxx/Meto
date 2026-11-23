#include <metal_stdlib>
using namespace metal;

kernel void transpose(device const float *a, device float *out,
                      constant uint &rows, constant uint &cols,
                      uint2 tileID [[threadgroup_position_in_grid]],
                      uint2 local [[thread_position_in_threadgroup]]) {
    // A 32x8 group moves a 32x32 tile. Padding separates the starts of adjacent
    // rows when threads later read down a column in threadgroup memory.
    threadgroup float tile[32][33];
    uint x = tileID.x * 32 + local.x;
    for (uint dy = local.y; dy < 32; dy += 8) {
        uint y = tileID.y * 32 + dy;
        if (x < cols && y < rows) tile[dy][local.x] = a[y*cols + x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Swap both the tile coordinates and its internal axes. Adjacent local.x
    // lanes now write adjacent addresses, just as they did on the input read.
    uint dstX = tileID.y * 32 + local.x;
    for (uint dy = local.y; dy < 32; dy += 8) {
        uint dstY = tileID.x * 32 + dy;
        if (dstX < rows && dstY < cols) out[dstY*rows + dstX] = tile[local.x][dy];
    }
}
