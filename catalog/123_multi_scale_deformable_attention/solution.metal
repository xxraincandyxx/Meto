#include <metal_stdlib>
using namespace metal;

kernel void interpolate(device const float *values, device const int2 *shape,
                        device const float2 *locations, device const float *weights,
                        device float *out, constant uint &heads, constant uint &d,
                        constant uint &levels, constant uint &points,
                        uint idx [[thread_position_in_grid]]) {
    uint feature = idx % d, query_head = idx / d, head = query_head % heads;
    float sum = 0;
    uint offset = 0;
    for (uint l = 0; l < levels; ++l) {
        int h = shape[l].x, w = shape[l].y;
        for (uint p = 0; p < points; ++p) {
            uint sample = (query_head * levels + l) * points + p;
            // Normalized positions refer to pixel centers. Convert to grid
            // space, then distribute this sample over its four neighbors.
            float2 pixel = locations[sample] * float2(w, h) - 0.5f;
            int2 base = int2(floor(pixel));
            float2 fraction = pixel - float2(base);
            for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
                int2 pos = base + int2(x, y);
                if (pos.x < 0 || pos.x >= w || pos.y < 0 || pos.y >= h) continue;
                float coeff = (x ? fraction.x : 1 - fraction.x)
                            * (y ? fraction.y : 1 - fraction.y);
                uint row = offset + pos.y * w + pos.x;
                sum += weights[sample] * coeff * values[(row * heads + head) * d + feature];
            }
        }
        offset += h * w;  // Levels occupy consecutive regions of the value table.
    }
    out[idx] = sum;
}
