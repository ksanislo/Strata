// include/strata/program/slot_persist.hpp - --session-dir with --batch: which slot conversation is saved before its
// slot is overwritten (header-only: it compares token lists).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::program {

/// `ids` starts with `head` (or is it).
inline bool slot_ids_continue(const std::vector<int32_t>& ids, const std::vector<int32_t>& head) {
    return head.size() <= ids.size() && std::equal(head.begin(), head.end(), ids.begin());
}

/// A slot's conversation `held` is saved before `next` overwrites the slot, unless it is not kept (`cached` false:
/// running, tainted, pictures, half written), the request writing the slot continues from this slot
/// (`from_this_slot`: the same conversation, perhaps from its turn checkpoint), `next` continues it, or one of
/// `others` (the other slots that will be saved in their turn) holds it or more.
inline bool slot_persist_wanted(bool cached, bool from_this_slot, const std::vector<int32_t>& held,
                                const std::vector<int32_t>& next,
                                const std::vector<const std::vector<int32_t>*>& others) {
    if (!cached || held.empty() || from_this_slot) return false;
    if (slot_ids_continue(next, held)) return false;
    for (const std::vector<int32_t>* o : others)
        if (o != nullptr && slot_ids_continue(*o, held)) return false;
    return true;
}

} // namespace strata::program
