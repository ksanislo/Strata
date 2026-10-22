#pragma once
// STRATA_ELASTIC_IDLE: how long a --batch slot has kept the same conversation idle.  Pure (slot_idle_test).

#include <cstddef>

namespace strata::program {

/// One slot as last seen: `stamp` names the request it served last (its start time), `n` the tokens it holds.  An idle
/// stretch starts whenever the slot is not keeping a conversation, or keeps another one (stamp or length changed).
template<class Time>
struct SlotIdleSeen {
    Time stamp{}, since{};
    size_t n = 0;
    bool kept = false;

    /// Records the slot at `now`; true when it has kept the same conversation since `since` (the stretch goes on).
    bool see(Time now, bool keeps, Time stamp_now, size_t n_now) {
        if (keeps && kept && stamp == stamp_now && n == n_now) return true;
        stamp = stamp_now;
        since = now;
        n = n_now;
        kept = keeps;
        return false;
    }
};

}  // namespace strata::program
