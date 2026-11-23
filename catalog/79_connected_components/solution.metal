#include <metal_stdlib>
using namespace metal;

// Parent links always point to smaller IDs. Following this chain reaches a
// root, and the monotonic direction prevents cycles among racing edge lanes.
inline int root(device atomic_int* parent, int x) {
    int next = atomic_load_explicit(parent + x, memory_order_relaxed);
    while (x != next) { x = next; next = atomic_load_explicit(parent + x, memory_order_relaxed); }
    return x;
}

kernel void unite(device const int* source, device const int* dest,
                  device atomic_int* parent, uint edge [[thread_position_in_grid]]) {
    int a = source[edge], b = dest[edge];
    for (;;) {
        a = root(parent, a); b = root(parent, b);
        if (a == b) return;
        int high = max(a, b), low = min(a, b), expected = high;
        // Only replace a self-parent. If another edge already replaced it,
        // follow both chains again so neither union is lost.
        if (atomic_compare_exchange_weak_explicit(parent + high, &expected, low,
                    memory_order_relaxed, memory_order_relaxed)) return;
    }
}

kernel void compress(device atomic_int* parent, uint node [[thread_position_in_grid]]) {
    atomic_store_explicit(parent + node, root(parent, int(node)), memory_order_relaxed);
}
