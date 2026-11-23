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

kernel void local_bfs(device const int* grid, device int* result,
                      constant uint& rows, constant uint& cols, constant uint& source,
                      constant uint& target, uint t [[thread_index_in_threadgroup]]) {
    // A 64x64 grid fits entirely in one cooperating group's scratch space.
    // The queue holds each cell at most once; 128 bit masks cover all cells.
    threadgroup ushort queue[4096];
    threadgroup atomic_uint visited[128], tail, found;
    atomic_store_explicit(visited + t, 0u, memory_order_relaxed);
    if (t == 0) {
        queue[0] = ushort(source);
        atomic_store_explicit(&tail, 1u, memory_order_relaxed);
        atomic_store_explicit(&found, uint(source == target), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) atomic_fetch_or_explicit(visited + source / 32, 1u << (source % 32), memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint first = 0, last = 1, level = 0;
    while (first < last && !atomic_load_explicit(&found, memory_order_relaxed)) {
        // The group must agree to enter this level before anyone marks the goal.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint pos = first + t; pos < last; pos += 128) {
            uint cell = queue[pos], x = cell % cols, y = cell / cols;
            for (uint direction = 0; direction < 4; ++direction) {
                if ((direction == 0 && !x) || (direction == 1 && x + 1 == cols) ||
                    (direction == 2 && !y) || (direction == 3 && y + 1 == rows)) continue;
                int delta = direction == 0 ? -1 : direction == 1 ? 1 : direction == 2 ? -int(cols) : int(cols);
                uint neighbor = uint(int(cell) + delta);
                if (grid[neighbor] != 0) continue;
                uint mask = 1u << (neighbor % 32);
                uint old = atomic_fetch_or_explicit(visited + neighbor / 32, mask, memory_order_relaxed);
                if (!(old & mask)) {
                    uint slot = atomic_fetch_add_explicit(&tail, 1u, memory_order_relaxed);
                    queue[slot] = ushort(neighbor);
                    if (neighbor == target) atomic_store_explicit(&found, 1u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        first = last;
        last = atomic_load_explicit(&tail, memory_order_relaxed);
        ++level;
    }
    if (!t) result[0] = atomic_load_explicit(&found, memory_order_relaxed) ? int(level) : -1;
}
