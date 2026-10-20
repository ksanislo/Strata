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
// conversations decode, its MTP draft right after it; the host decides each slot's accepted prefix once the last stage
// has picked, and the group's next window waits for that commit and the slots' new drafts.

/// Whether windows carry drafts: at most `max_active` conversations decode (0 or less: never).
inline bool piped_mtp_on(int active_total, int max_active) { return max_active > 0 && active_total <= max_active; }

/// A group's hand-off rows start here: room for two rows per slot, so no two groups share a hand-off row.
inline int piped_mtp_hbase(int group, int slots_per_group) { return group * 2 * slots_per_group; }

/// The hand-off rows the pipelined groups need (at least `max_t`, the windows' own size).
inline int piped_mtp_handoff_rows(int batch, int max_t) { return 2 * batch > max_t ? 2 * batch : max_t; }

struct Layout {
    int S = 0;                          ///< rows
    int rows[8] = {};                   ///< row -> slot
    int32_t tok[8] = {};
    int64_t pos[8] = {};
    int first[8] = {};                  ///< group slot t's first row (-1: not in the window)
    int n[8] = {};                      ///< ... and its rows (1, or 2 with a draft)
    int active = 0;
    bool drafts = false;                ///< some slot has a draft row (its commit waits for the verdict)
};

/// The window of the group whose slots are [base, base + gs): live slots only, each its token (x at p), and its draft
/// at p + 1 when `on`, the slot has one, it fits in `max_rows` and the slot could keep both (p + 3 <= max_context).
inline Layout piped_mtp_layout(int base, int gs, const bool* active, const bool* draft_ready, const int32_t* x,
                               const int32_t* draft, const int64_t* p, bool on, int max_rows, int64_t max_context) {
    Layout L;
    if (max_rows > 8) max_rows = 8;
    for (int t = 0; t < gs && t < 8; ++t) {
        L.first[t] = -1;
        L.n[t] = 0;
        if (!active[t] || L.S >= max_rows) continue;
        L.first[t] = L.S;
        L.rows[L.S] = base + t;
        L.tok[L.S] = x[t];
        L.pos[L.S] = p[t];
        ++L.S;
        if (on && draft_ready[t] && L.S < max_rows && p[t] + 3 <= max_context) {
            L.rows[L.S] = base + t;
            L.tok[L.S] = draft[t];
            L.pos[L.S] = p[t] + 1;
            ++L.S;
            L.drafts = true;
        }
        L.n[t] = L.S - L.first[t];
        ++L.active;
    }
    return L;
}

/// A slot's accepted rows: 2 when it had a draft row, the target picked that draft after its token, and both tokens
/// can be emitted; else 1 (as --batch-mtp's serial windows decide).
inline int piped_mtp_keep(int n_rows, int32_t pick0, int32_t draft, bool pick0_eos, bool stop, int64_t produced,
                          int64_t max_new, int64_t p, int64_t max_context) {
    return n_rows >= 2 && pick0 == draft && !pick0_eos && !stop && produced + 2 <= max_new && p + 3 <= max_context ? 2
                                                                                                                  : 1;
}

}  // namespace strata::program::batch_groups
