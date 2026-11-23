#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a,
    device const float* b,
    device float* merged,
    constant uint& m,
    constant uint& n,
    uint i [[thread_position_in_grid]]) {
    // The destination equals this element's rank in its own list plus its
    // insertion rank in the other list. On equal keys, A precedes B; this
    // asymmetric comparison gives every element its own destination index.
    bool fromA = i < m;
    uint own = fromA ? i : i - m;
    float key = fromA ? a[own] : b[own];
    device const float* other = fromA ? b : a;
    uint lo = 0, hi = fromA ? n : m;
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        if (other[mid] < key || (!fromA && other[mid] == key)) lo = mid + 1;
        else hi = mid;
    }
    merged[own + lo] = key;
}
