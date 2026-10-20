#pragma once
// The elastic --batch slot sessions (STRATA_ELASTIC_SLOTS): where a slot state's growing arrays sit in its VMM range
// and how many of their chunks a context of `cells` cells needs.  Pure arithmetic (slot_sizing_test).

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {

struct SlotArray {
    uint64_t per_unit = 0;   ///< bytes per unit: a K/V page slot, or one indexer pooled row
    int64_t units = 0;       ///< units at the full context
    bool kv = true;          ///< the units are K/V page slots (else pooled rows)
    uint64_t off = 0;        ///< offset in the range, whole chunks (slot_layout)
};

/// Only an array larger than one chunk is worth a range of its own; the rest are carved as before.
inline bool slot_array_elastic(uint64_t full_bytes, uint64_t gran) { return gran > 0 && full_bytes > gran; }

/// Puts each array at a chunk boundary, in order; returns the range's bytes.
inline uint64_t slot_layout(std::vector<SlotArray>& arrays, uint64_t gran) {
    uint64_t at = 0;
    for (SlotArray& a : arrays) {
        a.off = at;
        at += ((uint64_t) a.units * a.per_unit + gran - 1) / gran * gran;
    }
    return at;
}

/// Units an array needs for `cells` cells, at least one (an idle slot's pad row writes position 0).  After a reset the
/// K/V streaming map hands out page slots 0, 1, 2.. in order and evicts only once all are taken, so a slot holding
/// `cells` cells uses at most ceil(cells / page_size) of them; the indexer writes row pos / idx_block and the spare after it.
inline int64_t slot_units(const SlotArray& a, int64_t cells, int64_t page_size, int64_t idx_block) {
    cells = std::max<int64_t>(cells, 1);
    const int64_t u = a.kv ? (cells + page_size - 1) / page_size : cells / idx_block + 2;
    return std::min<int64_t>(a.units, u);
}

/// Chunks from the array's start that hold `units` units.
inline int64_t slot_chunks(const SlotArray& a, int64_t units, uint64_t gran) {
    return (int64_t) (((uint64_t) units * a.per_unit + gran - 1) / gran);
}

/// Whole units the first `chunks` mapped chunks of an array hold.
inline int64_t slot_units_in(const SlotArray& a, int64_t chunks, uint64_t gran) {
    return std::min<int64_t>(a.units, (int64_t) ((uint64_t) chunks * gran / a.per_unit));
}

/// Cells a growth to `need` cells asks for: whole steps past it, so the driver is called once per `step` cells.
inline int64_t slot_grow_target(int64_t need, int64_t step, int64_t max_cells) {
    if (step <= 0) return std::min(need, max_cells);
    return std::min<int64_t>(max_cells, (need + step - 1) / step * step + step);
}

}  // namespace strata::core
