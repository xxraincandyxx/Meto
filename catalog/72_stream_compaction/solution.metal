#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x,
    device const int* rank,
    device float* out,
    constant uint& count,
    uint i [[thread_position_in_grid]]) {
    // A prefix count gives every survivor an increasing, unique destination
    // without locks. rank[count - 1] is the total number of positives, so
    // the zero-fill and scatter below always write disjoint ranges.
    if (i >= uint(rank[count - 1])) out[i] = 0;
    if (x[i] > 0) out[rank[i] - 1] = x[i];
}
