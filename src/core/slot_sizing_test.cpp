// The elastic slot sessions' arithmetic (slot_sizing.hpp): layout at chunk boundaries, units and chunks per context.
#include "strata/core/slot_sizing.hpp"

#include <cstdio>

using namespace strata::core;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

int main() {
    const uint64_t G = 2ull << 20;
    // the live shape: 262144 cells, 4-cell pages, KV streaming with 8192 page slots; INT8 codes 2 heads x 4 cells x 256 B
    // per page slot, scales 2 x 4 x 8 B; the indexer 128 floats per row, a row per 4 cells (+2)
    const int64_t page = 4, idx_block = 4, n_slots = 8192, pooled_rows = 262144 / idx_block + 2;
    const uint64_t codes = 2 * 4 * 256, scales = 2 * 4 * 8;
    check(slot_array_elastic(n_slots * codes, G), "16 MiB of codes are elastic");
    check(!slot_array_elastic(n_slots * scales, G), "512 KiB of scales stay carved");
    check(!slot_array_elastic(G, G), "exactly one chunk stays carved");
    check(!slot_array_elastic(1ull << 40, 0), "no VMM: nothing elastic");

    std::vector<SlotArray> a = {{codes, n_slots, true, 0}, {codes, n_slots, true, 0}, {128 * 4, pooled_rows, false, 0}};
    const uint64_t total = slot_layout(a, G);
    check(a[0].off == 0 && a[1].off == 16 * G / 2 && a[2].off == 16 * G, "arrays at chunk boundaries, in order");
    check(total == 16 * G + (uint64_t) ((pooled_rows * 512 + G - 1) / G) * G, "range bytes");
    for (const SlotArray& x : a) check(x.off % G == 0, "chunk aligned");

    // units: at least one, page slots by ceil(cells / page), capped; pooled rows cells / block + 2, capped
    check(slot_units(a[0], 0, page, idx_block) == 1, "an empty slot keeps one page slot (pad rows)");
    check(slot_units(a[0], 1, page, idx_block) == 1, "one cell, one page slot");
    check(slot_units(a[0], 5, page, idx_block) == 2, "five cells, two page slots");
    check(slot_units(a[0], 1 << 20, page, idx_block) == n_slots, "page slots capped at n_slots");
    check(slot_units(a[2], 0, page, idx_block) == 2, "pooled rows: the spare after row 0");
    check(slot_units(a[2], 4096, page, idx_block) == 1026, "pooled rows at 4096 cells");
    check(slot_units(a[2], 1 << 30, page, idx_block) == pooled_rows, "pooled rows capped");

    // chunks: 4096 cells of codes are exactly one chunk; one more cell takes a second
    check(slot_chunks(a[0], slot_units(a[0], 4096, page, idx_block), G) == 1, "4096 cells = 1 chunk of codes");
    check(slot_chunks(a[0], slot_units(a[0], 4097, page, idx_block), G) == 2, "4097 cells = 2 chunks");
    check(slot_chunks(a[2], slot_units(a[2], 1, page, idx_block), G) == 1, "pooled: one chunk for the first rows");
    check(slot_units_in(a[0], 1, G) == 1024, "a chunk holds 1024 page slots");
    check(slot_units_in(a[0], 100, G) == n_slots, "held units capped");
    check(slot_units_in(a[2], 0, G) == 0, "nothing mapped, nothing held");

    // growth targets: whole steps with a step to spare, never past the context
    check(slot_grow_target(1, 4096, 262144) == 8192, "first growth");
    check(slot_grow_target(4096, 4096, 262144) == 8192, "on a step boundary");
    check(slot_grow_target(4097, 4096, 262144) == 12288, "past a boundary");
    check(slot_grow_target(262000, 4096, 262144) == 262144, "capped at the context");
    check(slot_grow_target(100, 0, 262144) == 100, "no step: just the need");

    if (failures == 0) std::printf("slot_sizing_test: all passed\n");
    return failures == 0 ? 0 : 1;
}
