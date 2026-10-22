// include/strata/program/fast_lane.hpp - STRATA_FAST_LANE: a short prompt read through the pipelined decode windows.
//
// A slot whose prompt has at most STRATA_FAST_LANE_MAX tokens to read is not queued on the prompt path (where it waits
// for the big chunk in progress on every stage): its prompt tokens go through its group's windows as rows that are all
// kept, as drafts that are all accepted (the deferred commit keeps every row).  The window holding the prompt's last
// token gives the first token (that row's pick).  Pure (no GPU, no environment): generate.cpp passes in what it knows.
#pragma once

#include "strata/program/batch_groups.hpp"

#include <cstdint>

namespace strata::program::fast_lane {

/// A read of an `n`-token prompt whose first `resume` tokens the slot holds goes through the lane: at most `max`
/// tokens are read before the last one (whose row gives the first token).
inline bool eligible_len(int64_t n, int64_t resume, int64_t max) {
    return max > 0 && n >= 1 && resume >= 0 && resume <= n - 1 && n - 1 - resume <= max;
}

/// The window of the group whose slots are [base, base + gs), as batch_groups::piped_mtp_layout, where `lane_left[t]`
/// > 0 marks a lane slot with that many prompt rows still to feed (its token at p[t] and the next ones, `lane_src[t]`
/// pointing at the prompt's token p[t]).  Every live slot gets one row; then the lane slots, in order, as many of the
/// rows left as they can use (at most the context's end); the other slots' drafts share what remains, one per slot in
/// turn, as piped_mtp_layout does.
inline batch_groups::Layout layout(int base, int gs, const bool* active, const int64_t* lane_left,
                                   const int32_t* const* lane_src, const int* n_draft, const int32_t* x,
                                   const int32_t* draft, const int64_t* p, const int64_t* left, bool on,
                                   int max_drafts, int max_rows, int64_t max_context) {
    batch_groups::Layout L;
    if (max_rows > 8) max_rows = 8;
    if (gs > 8) gs = 8;
    int live = 0, want[8] = {}, give[8] = {};
    for (int t = 0; t < gs; ++t) {
        if (!active[t]) continue;
        ++live;
        int64_t w;
        if (lane_left[t] > 0) {
            w = lane_left[t] - 1;
            if (w > max_context - p[t] - 1) w = max_context - p[t] - 1;   // row p + j needs p + j + 1 <= max_context
        } else {
            w = on ? (n_draft[t] < max_drafts ? n_draft[t] : max_drafts) : 0;
            if (w > left[t] - 1) w = left[t] - 1;
            if (w > max_context - p[t] - 2) w = max_context - p[t] - 2;
        }
        want[t] = w > 0 ? (int) w : 0;
    }
    int budget = max_rows - live;
    for (int t = 0; t < gs && budget > 0; ++t)
        if (active[t] && lane_left[t] > 0) {
            give[t] = want[t] < budget ? want[t] : budget;
            budget -= give[t];
        }
    for (int more = 1; budget > 0 && more;) {
        more = 0;
        for (int t = 0; t < gs && budget > 0; ++t)
            if (active[t] && lane_left[t] <= 0 && give[t] < want[t]) {
                ++give[t];
                --budget;
                more = 1;
            }
    }
    for (int t = 0; t < gs; ++t) {
        L.first[t] = -1;
        L.n[t] = 0;
        if (!active[t] || L.S >= max_rows) continue;
        const bool lane = lane_left[t] > 0;
        L.first[t] = L.S;
        L.rows[L.S] = base + t;
        L.tok[L.S] = lane ? lane_src[t][0] : x[t];
        L.pos[L.S] = p[t];
        ++L.S;
        for (int j = 0; j < give[t] && L.S < max_rows; ++j) {
            L.rows[L.S] = base + t;
            L.tok[L.S] = lane ? lane_src[t][j + 1] : draft[t * 8 + j];
            L.pos[L.S] = p[t] + 1 + j;
            ++L.S;
            L.drafts = true;   // (a slot with several rows: the commit waits for the verdict, which keeps them all)
        }
        L.n[t] = L.S - L.first[t];
        ++L.active;
    }
    return L;
}

/// A lane slot's window of `rows` rows from position `p` of an `n`-token prompt: every row is kept and the slot moves
/// to `p + rows`; `done` when the last row was the prompt's last token: its pick (row `pick_row`) is the first token,
/// at position n.
struct Step {
    int keep = 0;
    int64_t p = 0;
    bool done = false;
    int pick_row = -1;
};
inline Step step(int rows, int64_t p, int64_t n) {
    Step s;
    s.keep = rows;
    s.p = p + rows;
    s.done = s.p >= n;
    s.pick_row = s.done ? rows - 1 : -1;
    return s;
}

/// The tokens a slot drafter's catch-up pairs with the window's rows (row t: the token at p + t + 1): the prompt's next
/// tokens, and for the prompt's last row the first token (`pick`).  `ids` is the whole prompt (n tokens).
inline void catchup_next(const int32_t* ids, int64_t n, int64_t p, int rows, int32_t pick, int32_t* out) {
    for (int t = 0; t < rows; ++t) {
        const int64_t c = p + t + 1;
        out[t] = c < n ? ids[c] : pick;
    }
}

}  // namespace strata::program::fast_lane
