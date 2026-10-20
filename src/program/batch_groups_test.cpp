// src/program/batch_groups_test.cpp - how many pipelined slot groups --batch runs with (header-only, CPU).
//
//   1. `auto` picks the most groups, at most one per stage, that divide the slots;
//   2. a layer split with --batch >= 2 and no --batch-groups means `auto` (0.1.41);
//   3. --batch-mtp (or --adapt-async 1) turns that default into one group, and says which feature did it;
//   4. a number or `auto` given explicitly is honoured, with or without those features;
//   5. no layer split: the number given (1) stays.
#include "strata/program/batch_groups.hpp"
#include "strata/core/batch_rows.hpp"

#include <cstdio>
#include <initializer_list>

namespace bg = strata::program::batch_groups;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bg::Input in(int batch, int stages) {
    bg::Input i;
    i.batch = batch;
    i.later_stages = stages;
    return i;
}
}  // namespace

int main() {
    std::printf("batch_groups_test\n");
    check(bg::auto_groups(0, 4) == 1, "auto: no layer split, one group");
    check(bg::auto_groups(1, 2) == 2 && bg::auto_groups(1, 4) == 2 && bg::auto_groups(1, 3) == 1,
          "auto: two GPUs, 2 groups when the slots divide by 2");
    check(bg::auto_groups(3, 8) == 4 && bg::auto_groups(3, 6) == 3 && bg::auto_groups(3, 2) == 2,
          "auto: at most one group per GPU, and a divisor of the slots");

    {   // the 0.1.41 default
        const auto r = bg::resolve(in(2, 1));
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None, "layer split, --batch 2, nothing given: 2 groups");
    }
    {
        const auto r = bg::resolve(in(1, 1));
        check(r.groups == 1 && !r.from_auto, "--batch 1: no default");
    }
    {
        const auto r = bg::resolve(in(2, 0));
        check(r.groups == 1 && !r.from_auto, "no layer split: one group");
    }

    {   // the production shape: parallel 2 on a 2-GPU split with --batch-mtp
        bg::Input i = in(2, 1);
        i.batch_mtp = true;
        auto r = bg::resolve(i);
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None && r.mtp_dropped,
              "--batch-mtp, nothing given: the pipeline wins, --batch-mtp is dropped");
        i.mtp_serial = true;
        r = bg::resolve(i);
        check(r.groups == 1 && r.from_auto && r.serial == bg::Serial::BatchMtp && r.would_be == 2 && !r.mtp_dropped,
              "STRATA_BATCH_MTP_SERIAL: one group, because of --batch-mtp");
        check(bg::serial_what(r.serial)[0] == '-', "the reason names the feature");
    }
    {
        bg::Input i = in(2, 1);
        i.adapt_async = true;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.serial == bg::Serial::AdaptAsync, "--adapt-async 1, nothing given: one group");
    }
    {
        bg::Input i = in(2, 1);
        i.batch_mtp = true;
        i.adapt_async = true;
        check(bg::resolve(i).serial == bg::Serial::AdaptAsync, "both: --adapt-async 1 still gives one group");
        i.mtp_serial = true;
        check(bg::resolve(i).serial == bg::Serial::BatchMtp, "both with STRATA_BATCH_MTP_SERIAL: --batch-mtp is named");
    }
    {   // nothing to give way: the slots do not divide, so auto is one group anyway and there is nothing to say
        bg::Input i = in(3, 1);
        i.batch_mtp = true;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.serial == bg::Serial::None && r.would_be == 1, "auto would be 1 group anyway: no reason");
    }
    {   // explicit values are honoured
        bg::Input i = in(4, 1);
        i.set = true;
        i.given = 2;
        i.batch_mtp = true;
        i.adapt_async = true;
        const auto r = bg::resolve(i);
        check(r.groups == 2 && !r.from_auto && r.serial == bg::Serial::None, "--batch-groups 2 with --batch-mtp: 2");
        i.given = 1;
        check(bg::resolve(i).groups == 1, "--batch-groups 1: 1 (the opt-out)");
        i.given = 4;   // not checked here: generate.cpp refuses what does not divide
        check(bg::resolve(i).groups == 4, "--batch-groups 4 is returned as given");
    }
    {
        bg::Input i = in(2, 1);
        i.set = true;
        i.auto_given = true;
        i.batch_mtp = true;
        const auto r = bg::resolve(i);
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None,
              "--batch-groups auto with --batch-mtp: auto asked for the pipeline, 2 groups");
    }
    {
        bg::Input i = in(2, 0);
        i.set = true;
        i.given = 2;
        check(bg::resolve(i).groups == 2, "an explicit number without a split is returned (the engine warns)");
    }

    // ---- STRATA_PIPED_MTP
    {   // the default pipeline keeps --batch-mtp instead of dropping it
        bg::Input i = in(4, 3);
        i.batch_mtp = true;
        i.piped_mtp = true;
        const auto r = bg::resolve(i);
        check(r.groups == 4 && !r.mtp_dropped && r.serial == bg::Serial::None, "STRATA_PIPED_MTP: 4 groups, MTP kept");
        i.piped_mtp = false;
        check(bg::resolve(i).mtp_dropped, "without it the pipeline drops --batch-mtp");
    }
    check(bg::piped_mtp_on(1, 2) && bg::piped_mtp_on(2, 2) && !bg::piped_mtp_on(3, 2), "gate: at most 2 conversations");
    check(!bg::piped_mtp_on(0, 0) && bg::piped_mtp_on(4, 4), "gate: 0 is never, the number otherwise");
    check(bg::piped_mtp_hbase(0, 1) == 0 && bg::piped_mtp_hbase(3, 1) == 6 && bg::piped_mtp_hbase(1, 2) == 4,
          "hbase: two hand-off rows per slot of each group");
    check(bg::piped_mtp_handoff_rows(4, 8) == 8 && bg::piped_mtp_handoff_rows(8, 8) == 16,
          "hand-off rows: 2 per slot, at least the window");
    {   // groups never share a hand-off row: [hbase, hbase + S) disjoint for S <= 2 * gs, all inside the hand-off
        for (int batch : {2, 4, 6, 8, 12, 16})
            for (int groups : {2, 3, 4})
                if (batch % groups == 0) {
                    const int gs = batch / groups, rows = bg::piped_mtp_handoff_rows(batch, 8);
                    for (int gi = 0; gi < groups; ++gi)
                        check(bg::piped_mtp_hbase(gi, gs) + 2 * gs <= rows &&
                              (gi == 0 || bg::piped_mtp_hbase(gi - 1, gs) + 2 * gs <= bg::piped_mtp_hbase(gi, gs)),
                              "hand-off ranges disjoint and in range");
                }
    }
    {
        const bool act[4] = {true, false, true, true}, rdy[4] = {true, true, false, true};
        const int32_t x[4] = {10, 11, 12, 13}, dr[4] = {20, 21, 22, 23};
        const int64_t p[4] = {100, 0, 200, 300};
        auto L = bg::piped_mtp_layout(4, 4, act, rdy, x, dr, p, true, 8, 1000);
        check(L.S == 5 && L.active == 3 && L.drafts, "layout: 2 + 1 + 2 rows, idle slot skipped");
        check(L.rows[0] == 4 && L.rows[1] == 4 && L.rows[2] == 6 && L.rows[3] == 7 && L.rows[4] == 7, "layout: row slots");
        check(L.tok[0] == 10 && L.tok[1] == 20 && L.pos[1] == 101 && L.tok[2] == 12 && L.tok[4] == 23 && L.pos[4] == 301,
              "layout: token then draft at p + 1");
        check(L.first[0] == 0 && L.n[0] == 2 && L.first[1] == -1 && L.first[2] == 2 && L.n[2] == 1 && L.first[3] == 3,
              "layout: each slot's rows");
        const char* why = strata::core::batch_rows_error(L.rows, L.S, L.pos, 8, 8, 8, true, 16);
        check(why == nullptr, "layout: a valid batch row layout");
        L = bg::piped_mtp_layout(4, 4, act, rdy, x, dr, p, false, 8, 1000);
        check(L.S == 3 && !L.drafts, "layout off: one row per live slot");
        L = bg::piped_mtp_layout(4, 4, act, rdy, x, dr, p, true, 4, 1000);
        check(L.S == 4 && L.n[0] == 2 && L.n[2] == 1 && L.n[3] == 1, "layout: the rows cap drops a draft, not a slot");
        L = bg::piped_mtp_layout(4, 4, act, rdy, x, dr, p, true, 8, 302);
        check(L.n[3] == 1 && L.n[0] == 2, "layout: no draft where the slot could not keep two");
    }
    check(bg::piped_mtp_keep(2, 5, 5, false, false, 10, 100, 50, 1000) == 2, "keep: draft picked");
    check(bg::piped_mtp_keep(2, 5, 6, false, false, 10, 100, 50, 1000) == 1, "keep: draft rejected");
    check(bg::piped_mtp_keep(1, 5, 5, false, false, 10, 100, 50, 1000) == 1, "keep: no draft row");
    check(bg::piped_mtp_keep(2, 5, 5, true, false, 10, 100, 50, 1000) == 1, "keep: eos first");
    check(bg::piped_mtp_keep(2, 5, 5, false, true, 10, 100, 50, 1000) == 1, "keep: stopped");
    check(bg::piped_mtp_keep(2, 5, 5, false, false, 99, 100, 50, 1000) == 1, "keep: one token left");
    check(bg::piped_mtp_keep(2, 5, 5, false, false, 10, 100, 998, 1000) == 1, "keep: at the context's end");

    std::printf("%s\n", g_fail == 0 ? "batch_groups_test: all passed" : "batch_groups_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
