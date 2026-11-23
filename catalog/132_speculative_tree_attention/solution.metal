#include <metal_stdlib>
using namespace metal;

kernel void lineage(device const int *parents, device bool *mask,
                    constant uint &prefix, constant uint &nodes, uint node [[thread_position_in_grid]]) {
    device bool *tree_row = mask + node * (prefix + nodes) + prefix;
    for (uint j = 0; j < nodes; ++j) tree_row[j] = false;
    // Follow this candidate's path to its root. Different candidates own
    // disjoint rows, so even parents shared by several paths need no atomics.
    for (int j = int(node); j >= 0; j = parents[j]) tree_row[j] = true;
}
