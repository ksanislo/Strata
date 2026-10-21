// include/strata/program/batch_groups.hpp - how many pipelined slot groups `--batch` runs with.
//
// `--batch-groups N` pipelines N groups of slots through the stages of a layer split; `--batch-groups auto` picks the
// most groups (at most one per stage) that divide the slots.  Since 0.1.41 a layer split with `--batch` >= 2 and no
// `--batch-groups` at all means `auto`.  The pipelined path does not run `--batch-mtp`'s drafts or the batch
// adaptive tier (`--adapt-async 1` between batch windows).  `--adapt-async 1` with the number of groups NOT given still
// makes `auto` resolve to one group (and says why).  `--batch-mtp` does not any more (0.1.43.1): on 2x A4000 and 2x RTX
// 3090 four concurrent streams lost 25-30% with the serial MTP windows against the pipelined default, so the default
// pipelines and `--batch-mtp` is left off there (said at start); STRATA_BATCH_MTP_SERIAL=1 or `--batch-groups 1` keeps
// the serial windows with the slots' drafts (the 0.1.42/0.1.43 behaviour).  A number or `auto` given on the command line
// is honoured as before.
//
// STRATA_PIPED_MTP=1 (opt-in): the pipelined groups keep --batch-mtp's drafts (piped_mtp_* below); --batch-mtp then
// stays on beside the pipeline instead of giving way to it.
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.
#pragma once

#include <cstdint>

namespace strata::program::batch_groups {

/// What kept `auto` from pipelining.
enum class Serial { None, BatchMtp, AdaptAsync };

struct Input {
    bool set = false;          ///< --batch-groups was given (a number or `auto`)
    bool auto_given = false;   ///< ... as `auto`
    int given = 1;             ///< the number given (1 when none)
    int batch = 0;             ///< the slots
    int later_stages = 0;      ///< the layer split's stages after CUDA0 (0: no layer split)
    bool batch_mtp = false;    ///< --batch-mtp is on (it passed the gates that do not concern the groups)
    bool adapt_async = false;  ///< --adapt-async 1 was asked for
    bool mtp_serial = false;   ///< STRATA_BATCH_MTP_SERIAL=1: --batch-mtp with no --batch-groups runs one serial group (0.1.43)
    bool piped_mtp = false;    ///< STRATA_PIPED_MTP=1: the pipelined groups run --batch-mtp's drafts
};

struct Result {
    int groups = 1;
    bool from_auto = false;            ///< the number came from `auto` (given or the 0.1.41 default)
    Serial serial = Serial::None;      ///< `auto` would have pipelined, but this feature needs the serial windows
    int would_be = 1;                  ///< what `auto` would have picked without it (== groups unless `serial`)
    bool mtp_dropped = false;          ///< --batch-mtp met the default pipeline: the pipeline wins, --batch-mtp is off
};

/// The most groups, at most one per stage of the layer split (`later_stages` + 1 GPUs), that divide `batch`.
inline int auto_groups(int later_stages, int batch) {
    int best = 1;
    if (later_stages <= 0) return best;
    for (int d = 2; d <= later_stages + 1 && d <= batch; ++d)
        if (batch % d == 0) best = d;
    return best;
}

inline Result resolve(const Input& in) {
    Result r;
    r.groups = in.given;
    r.would_be = in.given;
    const bool defaulted = !in.set && in.later_stages > 0 && in.batch > 1;   // the 0.1.41 default
    if (!in.auto_given && !defaulted) return r;
    r.from_auto = true;
    r.would_be = auto_groups(in.later_stages, in.batch);
    r.groups = r.would_be;
    if (!in.set && r.would_be > 1) {   // only the default gives way; `--batch-groups auto` asked for the pipeline
        if (in.batch_mtp && in.mtp_serial) r.serial = Serial::BatchMtp;
        else if (in.adapt_async) r.serial = Serial::AdaptAsync;
        r.mtp_dropped = in.batch_mtp && r.serial == Serial::None && !in.piped_mtp;
        if (r.serial != Serial::None) r.groups = 1;
    }
    return r;
}

inline const char* serial_what(Serial s) {
    return s == Serial::BatchMtp ? "--batch-mtp" : s == Serial::AdaptAsync ? "--adapt-async 1" : "";
}

// ---- STRATA_PIPED_MTP: a group's window carries each live slot's token and, while at most `max_active`
// conversations decode, up to `max_drafts` of its MTP drafts right after it; the host decides each slot's accepted
// prefix once the last stage has picked, and the group's next window waits for that commit and the slots' new drafts.

/// Whether windows carry drafts: at most `max_active` conversations decode (0 or less: never).
inline bool piped_mtp_on(int active_total, int max_active) { return max_active > 0 && active_total <= max_active; }

/// The most rows a group's window holds: each slot's token and `drafts` drafts, at most `max_t`.
inline int piped_mtp_group_rows(int slots_per_group, int drafts, int max_t) {
    const int r = slots_per_group * (1 + (drafts > 0 ? drafts : 0));
    return r < max_t ? r : max_t;
}

/// A group's hand-off rows start here, so no two groups share a hand-off row.
inline int piped_mtp_hbase(int group, int group_rows) { return group * group_rows; }

/// The hand-off rows the pipelined groups need (at least `max_t`, the windows' own size).
inline int piped_mtp_handoff_rows(int groups, int group_rows, int max_t) {
    return groups * group_rows > max_t ? groups * group_rows : max_t;
}

struct Layout {
    int S = 0;                          ///< rows
    int rows[8] = {};                   ///< row -> slot
    int32_t tok[8] = {};
    int64_t pos[8] = {};
    int first[8] = {};                  ///< group slot t's first row (-1: not in the window)
    int n[8] = {};                      ///< ... and its rows (1 + its drafts)
    int active = 0;
    bool drafts = false;                ///< some slot has a draft row (its commit waits for the verdict)
};

/// The window of the group whose slots are [base, base + gs): live slots only, each its token (x at p) and, when `on`,
/// its first drafts at p + 1 ...: at most `max_drafts` and the slot's own `n_draft[t]` (draft t*8 + j), no more than
/// it could keep (`left[t]` tokens to its max_new, the context's end), the rows past the tokens shared out one draft
/// per slot in turn.
inline Layout piped_mtp_layout(int base, int gs, const bool* active, const int* n_draft, const int32_t* x,
                               const int32_t* draft, const int64_t* p, const int64_t* left, bool on, int max_drafts,
                               int max_rows, int64_t max_context) {
    Layout L;
    if (max_rows > 8) max_rows = 8;
    if (gs > 8) gs = 8;
    int live = 0, want[8] = {}, give[8] = {};
    for (int t = 0; t < gs; ++t) {
        if (!active[t]) continue;
        ++live;
        int64_t w = on ? (n_draft[t] < max_drafts ? n_draft[t] : max_drafts) : 0;
        if (w > left[t] - 1) w = left[t] - 1;                       // keeping k rows emits k tokens
        if (w > max_context - p[t] - 2) w = max_context - p[t] - 2; // ... and needs p + k + 1 <= max_context
        want[t] = w > 0 ? (int) w : 0;
    }
    for (int budget = max_rows - live, more = 1; budget > 0 && more;) {
        more = 0;
        for (int t = 0; t < gs && budget > 0; ++t)
            if (give[t] < want[t]) {
                ++give[t];
                --budget;
                more = 1;
            }
    }
    for (int t = 0; t < gs; ++t) {
        L.first[t] = -1;
        L.n[t] = 0;
        if (!active[t] || L.S >= max_rows) continue;
        L.first[t] = L.S;
        L.rows[L.S] = base + t;
        L.tok[L.S] = x[t];
        L.pos[L.S] = p[t];
        ++L.S;
        for (int j = 0; j < give[t] && L.S < max_rows; ++j) {
            L.rows[L.S] = base + t;
            L.tok[L.S] = draft[t * 8 + j];
            L.pos[L.S] = p[t] + 1 + j;
            ++L.S;
            L.drafts = true;
        }
        L.n[t] = L.S - L.first[t];
        ++L.active;
    }
    return L;
}

/// A slot's accepted rows (1..n_rows): its token, then each draft the target picked after the row before it, while the
/// pick is not an end token and the tokens can be emitted (as --batch-mtp's serial windows decide for one draft).
/// `pick[j]` is row j's pick, `tok[j]` row j's token, `eos[j]` whether pick j ends the conversation.
inline int piped_mtp_keep(int n_rows, const int32_t* pick, const int32_t* tok, const bool* eos, bool stop,
                          int64_t produced, int64_t max_new, int64_t p, int64_t max_context) {
    if (stop) return 1;
    int k = 1;
    while (k < n_rows && pick[k - 1] == tok[k] && !eos[k - 1] && produced + k + 1 <= max_new && p + k + 2 <= max_context)
        ++k;
    return k;
}

}  // namespace strata::program::batch_groups
