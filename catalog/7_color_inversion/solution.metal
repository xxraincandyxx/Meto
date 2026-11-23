#include <metal_stdlib>
using namespace metal;

kernel void invert(device uchar4* pixels, uint i [[thread_position_in_grid]]) {
    // uchar4 loads one four-byte pixel. XOR with all ones computes 255 - x;
    // XOR with zero copies alpha. Each thread owns a disjoint four-byte word.
    pixels[i] ^= uchar4(255, 255, 255, 0);
}
