#include <metal_stdlib>
using namespace metal;

kernel void visit(device const int* grid, device const uint* frontier, device uint* next,
                  device atomic_uint* distance, device atomic_uint* tail,
                  constant uint& rows, constant uint& cols, constant uint& level,
                  uint i [[thread_position_in_grid]]) {
    uint cell = frontier[i];
    uint x = cell % cols, y = cell / cols;
    for (uint direction = 0; direction < 4; ++direction) {
        if ((direction == 0 && !x) || (direction == 1 && x + 1 == cols) ||
            (direction == 2 && !y) || (direction == 3 && y + 1 == rows)) continue;
        int offset = direction == 0 ? -1 : direction == 1 ? 1 : direction == 2 ? -int(cols) : int(cols);
        uint neighbor = uint(int(cell) + offset);
        if (grid[neighbor] != 0) continue;
        // All threads in this dispatch have the same BFS distance. Exactly
        // one wins the first visit and reserves a unique next-frontier slot.
        uint old = atomic_fetch_min_explicit(distance + neighbor, level, memory_order_relaxed);
        if (old == 0xffffffffu) {
            uint slot = atomic_fetch_add_explicit(tail, 1u, memory_order_relaxed);
            next[slot] = neighbor;
        }
    }
}
