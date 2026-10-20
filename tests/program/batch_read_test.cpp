// batch_read.hpp: the pieces of a --batch prompt read, where the concurrent read runs (synthetic: no model, no GPU).
#include "strata/program/batch_read.hpp"

#include <cstdio>
#include <cstring>
#include <iterator>
#include <vector>

using strata::program::batch_read_piece;
using strata::program::concurrent_read_off;
using strata::program::ConcurrentReadSetup;
using strata::program::piped_decode_budget_ms;
using strata::program::piped_read_lead_in;
using strata::program::piped_read_piece;
using strata::program::admit_turn_cut;
using strata::program::group_pads_reading;
using strata::program::slot_admit_direct;
using strata::program::slot_admit_off;
using strata::program::SlotAdmitSetup;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // no --batch, or the pipelined groups: one run
    CHECK(batch_read_piece(false, false, false, false, 2048, 8) == 0);
    CHECK(batch_read_piece(true, true, true, false, 2048, 8) == 0);
    // one GPU: a chunk per piece, whether or not a slot decodes (the yield point is the chunk boundary)
    CHECK(batch_read_piece(true, false, false, false, 2048, 8) == 2048);
    CHECK(batch_read_piece(true, false, false, true, 2048, 8) == 2048);
    // layer split with slots decoding: a chunk per piece as before
    CHECK(batch_read_piece(true, false, true, true, 2048, 8) == 2048);
    // layer split, nobody decoding: pieces of 8 chunks - the BYIELD of a shorter request can be taken (it was one run)
    CHECK(batch_read_piece(true, false, true, false, 2048, 8) == 8 * 2048);
    CHECK(batch_read_piece(true, false, true, false, 1024, 4) == 4096);
    // STRATA_BATCH_YIELD_CHUNKS=0 keeps the old single run
    CHECK(batch_read_piece(true, false, true, false, 2048, 0) == 0);
    // a degenerate chunk never gives a zero piece
    CHECK(batch_read_piece(true, false, false, false, 0, 8) == 1);
    // a piece is always a multiple of the chunk
    for (int64_t c : {256, 1000, 2048, 4096})
        for (int64_t k : {1, 3, 8, 16}) CHECK(batch_read_piece(true, false, true, false, c, k) % c == 0);
    // concurrent_read_off (STRATA_PIPED_CONCURRENT): runs with nothing set; each setting alone keeps it off, named
    CHECK(concurrent_read_off({}) == nullptr);
    bool ConcurrentReadSetup::*const each[] = {
        &ConcurrentReadSetup::prefill_help, &ConcurrentReadSetup::split_mtp_batch, &ConcurrentReadSetup::fs_slots,
        &ConcurrentReadSetup::pcie_dma,     &ConcurrentReadSetup::helper_caches,   &ConcurrentReadSetup::staged_source,
        &ConcurrentReadSetup::other_source};
    const char* named[std::size(each)] = {};
    auto same = [](const char* a, const char* b) { return a != nullptr && b != nullptr && std::strcmp(a, b) == 0; };
    for (size_t i = 0; i < std::size(each); ++i) {
        ConcurrentReadSetup s;
        s.*each[i] = true;
        named[i] = concurrent_read_off(s);
        CHECK(named[i] != nullptr);
        for (size_t j = 0; j < i; ++j) CHECK(!same(named[i], named[j]));   // a reason of its own
    }
    CHECK(same(named[0], "STRATA_PREFILL_HELP"));
    CHECK(same(named[3], "--pcie-mode dma"));
    // with several, the first in the struct's order is named
    for (size_t i = 0; i < std::size(each); ++i) {
        ConcurrentReadSetup s;
        for (size_t j = i; j < std::size(each); ++j) s.*each[j] = true;
        CHECK(same(concurrent_read_off(s), named[i]));
    }
    // piped_read_piece (#1906): off -> one run; whole chunks, at least one, at most 32
    CHECK(piped_read_piece(0, 8192) == 0);
    CHECK(piped_read_piece(-5, 8192) == 0);
    CHECK(piped_read_piece(32768, 8192) == 32768);
    CHECK(piped_read_piece(20000, 8192) == 16384);      // rounded down to whole chunks
    CHECK(piped_read_piece(4096, 8192) == 8192);        // never below one chunk
    CHECK(piped_read_piece(1 << 30, 8192) == 32 * 8192);
    CHECK(piped_read_piece(5000, 0) == 32);             // chunk 0 guarded as 1 (and the cap applies)
    for (int64_t c : {1024, 2048, 6144, 8192})
        for (int64_t p : {1, 1000, 9000, 50000, 300000}) CHECK(piped_read_piece(p, c) % c == 0);   // the header's rule
    // piped_decode_budget_ms: share x piece, capped, floored; negatives are 0
    CHECK(piped_decode_budget_ms(2000.0, 2.0, 400.0, 0.0) == 400.0);
    CHECK(piped_decode_budget_ms(2000.0, 0.35, 3000.0, 0.0) == 700.0);
    CHECK(piped_decode_budget_ms(2000.0, 0.0, 400.0, 150.0) == 150.0);
    CHECK(piped_decode_budget_ms(2000.0, 2.0, 0.0, 0.0) == 4000.0);   // no cap
    CHECK(piped_decode_budget_ms(-5.0, -1.0, 400.0, -3.0) == 0.0);
    // piped_read_lead_in: on, beside decoding slots, within 1000 ms + the budget of the previous piecewise read's end
    CHECK(piped_read_lead_in(true, true, 0.0, 400.0));
    CHECK(piped_read_lead_in(true, true, 1400.0, 400.0));
    CHECK(!piped_read_lead_in(true, true, 1401.0, 400.0));
    CHECK(!piped_read_lead_in(true, true, -1.0, 400.0));
    CHECK(!piped_read_lead_in(false, true, 0.0, 400.0));
    CHECK(!piped_read_lead_in(true, false, 0.0, 400.0));
    // slot_admit_off (STRATA_SLOT_ADMIT): needs the switch and the pipelined groups; each blocker named
    {
        SlotAdmitSetup s;
        CHECK(slot_admit_off(s) != nullptr);
        s.asked = true;
        CHECK(slot_admit_off(s) != nullptr);
        s.piped = true;
        CHECK(slot_admit_off(s) == nullptr);
        SlotAdmitSetup m = s; m.batch_mtp = true; CHECK(m.batch_mtp && slot_admit_off(m) != nullptr);
        SlotAdmitSetup c = s; c.cvec = true; CHECK(slot_admit_off(c) != nullptr);
        SlotAdmitSetup h = s; h.head_alias = true; CHECK(slot_admit_off(h) != nullptr);
        SlotAdmitSetup r = s; r.conc_off = "x"; CHECK(std::strcmp(slot_admit_off(r), "x") == 0);
    }
    // slot_admit_direct: beside decoding slots, text, unless another cache holds `gain` tokens more
    CHECK(!slot_admit_direct(false, false, 0, 0, 16384));      // nothing decodes: the old way stops nobody
    CHECK(!slot_admit_direct(true, true, 0, 0, 16384));        // pictures
    CHECK(slot_admit_direct(true, false, 0, 0, 16384));
    CHECK(slot_admit_direct(true, false, 1000, 17000, 16384));
    CHECK(!slot_admit_direct(true, false, 0, 16384, 16384));   // a promoted solo request: its main session holds it
    CHECK(slot_admit_direct(true, false, 0, 1 << 20, 0));      // gain 0: always beside
    // admit_turn_cut: the last turn token after resume; a short trailing turn of the tail role moves it back
    {
        const std::vector<int32_t> ids = {1, 9, 2, 3, 9, 4, 5, 9, 7, 6};
        CHECK(admit_turn_cut(ids, 0, 9, -1) == 7);
        CHECK(admit_turn_cut(ids, 7, 9, -1) == -1);            // none after resume
        CHECK(admit_turn_cut(ids, 0, -1, -1) == -1);
        CHECK(admit_turn_cut(ids, 0, 9, 7) == 7);              // the turn before it (4) is not the tail role
        const std::vector<int64_t> t = {1, 9, 2, 9, 4, 3, 9, 8};
        CHECK(admit_turn_cut(t, 0, 9, 7) == 6);                // turn 6's predecessor turn 3 is not the tail role
        const std::vector<int64_t> u = {1, 9, 2, 9, 7, 9, 8};
        CHECK(admit_turn_cut(u, 0, 9, 7) == 3);                // the turn before 5 is the tail role's: moved to 3
        CHECK(admit_turn_cut(u, 3, 9, 7) == 5);                // not past resume
    }
    // group_pads_reading: a reading slot below an active one would be a pad row
    {
        const bool a1[4] = {false, false, true, false}, r1[4] = {false, true, false, false};
        CHECK(group_pads_reading(a1, r1, 4));
        const bool a2[4] = {true, false, false, false}, r2[4] = {false, true, false, false};
        CHECK(!group_pads_reading(a2, r2, 4));
        const bool a3[1] = {false}, r3[1] = {true};
        CHECK(!group_pads_reading(a3, r3, 1));
    }
    if (fails == 0) std::printf("batch_read_test: ok\n");
    return fails == 0 ? 0 : 1;
}
