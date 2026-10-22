// src/program/fast_lane_test.cpp - STRATA_FAST_LANE's row bookkeeping (header-only, CPU): the lane slot's rows in a
// group window, the cursor, every row kept, the first token from the prompt's last row, the drafter's catch-up tokens.
#include "strata/program/fast_lane.hpp"
#include "strata/core/batch_rows.hpp"

#include <cstdio>
#include <vector>

namespace fl = strata::program::fast_lane;
namespace bg = strata::program::batch_groups;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-84s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool valid(const bg::Layout& L, int hbase = 0, int hrows = 32) {
    return strata::core::batch_rows_error(L.rows, L.S, L.pos, 8, 8, hbase, true, hrows) == nullptr;
}

// One lane read simulated through group windows: slot 0 reads `n` tokens from `resume` while slot 1 decodes (its
// pick at a row = its token + 1); the picks are a function of the row's token. Returns the slot's fed tokens, the
// windows it took and its first token.
struct Sim {
    std::vector<int32_t> fed;
    int windows = 0, rows_max = 0;
    int32_t first = -1;
    int64_t p_end = -1;
    bool rows_ok = true, keep_ok = true, partner_ok = true;
};
Sim simulate(int64_t n, int64_t resume, bool partner, bool drafts_on, int max_rows = 8) {
    std::vector<int32_t> ids((size_t) n);
    for (int64_t i = 0; i < n; ++i) ids[(size_t) i] = (int32_t) (1000 + i);
    Sim s;
    s.fed.assign(ids.begin(), ids.begin() + resume);
    int64_t p = resume;
    int64_t pp = 500;   // the partner's position
    int32_t px = 7;
    for (int guard = 0; guard < 1000; ++guard) {
        const bool act[2] = {true, partner};
        const int64_t lane_left[2] = {n - p, 0};
        const int32_t* src[2] = {ids.data() + p, nullptr};
        const int nd[2] = {0, 2};
        const int32_t x[2] = {ids[(size_t) p], px};
        int32_t dr[16] = {};
        dr[8] = px + 1;
        dr[9] = px + 2;
        const int64_t ps[2] = {p, pp}, left[2] = {1000, 1000};
        const bg::Layout L = fl::layout(4, 2, act, lane_left, src, nd, x, dr, ps, left, drafts_on, 2, max_rows, 1 << 20);
        s.rows_ok = s.rows_ok && valid(L, 8) && L.first[0] == 0 && L.S <= max_rows;
        const int rows = L.n[0];
        s.rows_max = rows > s.rows_max ? rows : s.rows_max;
        for (int j = 0; j < rows; ++j)
            s.rows_ok = s.rows_ok && L.rows[j] == 4 && L.tok[j] == ids[(size_t) (p + j)] && L.pos[j] == p + j;
        if (partner) s.partner_ok = s.partner_ok && L.first[1] == rows && L.n[1] >= 1 && L.tok[rows] == px && L.pos[rows] == pp;
        // the picks: row j's argmax = its token + 1 (the partner accepts nothing extra: one token a window)
        int32_t pick[8];
        for (int j = 0; j < L.S; ++j) pick[j] = L.tok[j] + 1;
        const fl::Step st = fl::step(rows, p, n);
        s.keep_ok = s.keep_ok && st.keep == rows;
        for (int j = 0; j < st.keep; ++j) s.fed.push_back(L.tok[j]);
        ++s.windows;
        if (partner) { px = pick[L.first[1]]; ++pp; }
        p = st.p;
        if (st.done) {
            s.first = pick[st.pick_row];
            s.p_end = p;
            break;
        }
    }
    return s;
}
}  // namespace

int main() {
    std::printf("fast_lane_test: STRATA_FAST_LANE's rows\n");
    check(fl::eligible_len(40, 0, 96), "eligible: 39 tokens before the last <= 96");
    check(fl::eligible_len(97, 0, 96) && !fl::eligible_len(98, 0, 96), "eligible: exactly at the cap (96 read, then the last)");
    check(fl::eligible_len(5000, 4950, 96), "eligible: a long prompt the slot holds all but 49 of");
    check(!fl::eligible_len(40, 0, 0), "eligible: off (max 0)");
    check(fl::eligible_len(40, 39, 96), "eligible: only the last token left (one row)");
    check(!fl::eligible_len(40, 40, 96), "eligible: nothing left to read is not a lane read");

    {   // the layout alone
        std::vector<int32_t> ids(40);
        for (int i = 0; i < 40; ++i) ids[(size_t) i] = 100 + i;
        const bool one[2] = {true, false}, two[2] = {true, true};
        const int64_t ll[2] = {40, 0};
        const int32_t* src[2] = {ids.data(), nullptr};
        const int nd[2] = {0, 2};
        const int32_t x[2] = {100, 7};
        int32_t dr[16] = {};
        dr[8] = 8;
        dr[9] = 9;
        const int64_t p[2] = {0, 300}, left[2] = {100, 100};
        bg::Layout L = fl::layout(0, 2, one, ll, src, nd, x, dr, p, left, true, 2, 8, 1 << 20);
        check(L.S == 8 && L.n[0] == 8 && L.first[1] == -1 && L.drafts && valid(L), "layout: a lone lane slot takes 8 rows");
        check(L.tok[0] == 100 && L.tok[7] == 107 && L.pos[7] == 7, "layout: the prompt's tokens at consecutive positions");
        L = fl::layout(0, 2, two, ll, src, nd, x, dr, p, left, true, 2, 8, 1 << 20);
        check(L.n[0] == 7 && L.n[1] == 1 && L.tok[7] == 7 && L.pos[7] == 300 && valid(L),
              "layout: beside a decoding slot: 7 lane rows, the partner's token, no drafts for it");
        L = fl::layout(0, 2, two, ll, src, nd, x, dr, p, left, false, 2, 6, 1 << 20);
        check(L.n[0] == 5 && L.n[1] == 1 && valid(L, 0, 6), "layout: 6-row hand-off: 5 lane rows + the partner");
        const int64_t ll3[2] = {3, 0};
        L = fl::layout(0, 2, two, ll3, src, nd, x, dr, p, left, true, 2, 8, 1 << 20);
        check(L.n[0] == 3 && L.n[1] == 3 && L.tok[4] == 8 && L.tok[5] == 9,
              "layout: a lane slot with 3 rows left; the partner's drafts take the rest");
        L = fl::layout(0, 2, one, ll, src, nd, x, dr, p, left, true, 2, 8, 5);
        check(L.n[0] == 5, "layout: no lane row past the context (p + j + 1 <= max)");
        const bool lane2[2] = {true, true};
        const int64_t llb[2] = {10, 10};
        const int32_t* src2[2] = {ids.data(), ids.data() + 20};
        const int64_t p2[2] = {0, 20};
        L = fl::layout(0, 2, lane2, llb, src2, nd, x, dr, p2, left, true, 2, 8, 1 << 20);
        check(L.n[0] == 7 && L.n[1] == 1 && L.tok[7] == 120 && valid(L), "layout: two lane slots: the first one first");
        const int64_t none[2] = {0, 0};
        const bg::Layout A = fl::layout(0, 2, two, none, src, nd, x, dr, p, left, true, 2, 8, 1 << 20);
        const bg::Layout B = bg::piped_mtp_layout(0, 2, two, nd, x, dr, p, left, true, 2, 8, 1 << 20);
        bool same = A.S == B.S && A.drafts == B.drafts;
        for (int r = 0; r < A.S && same; ++r) same = A.rows[r] == B.rows[r] && A.tok[r] == B.tok[r] && A.pos[r] == B.pos[r];
        check(same, "layout: without lane slots it is piped_mtp_layout's");
    }

    {   // the cursor and the first token, window after window
        Sim s = simulate(40, 0, false, true);
        check(s.windows == 5 && s.rows_max == 8 && s.rows_ok && s.keep_ok, "read 40 alone: 5 windows of 8 rows, every row kept");
        bool fed = (int) s.fed.size() == 40;
        for (int i = 0; i < 40 && fed; ++i) fed = s.fed[(size_t) i] == 1000 + i;
        check(fed, "read 40 alone: the slot holds the prompt's 40 tokens in order");
        check(s.first == 1000 + 39 + 1 && s.p_end == 40, "read 40 alone: first token = the last row's pick, at position 40");
        s = simulate(40, 0, true, true);
        check(s.windows == 6 && s.rows_max == 7 && s.rows_ok && s.partner_ok && s.keep_ok,
              "read 40 beside a decoding partner: 6 windows of up to 7 rows, the partner's row each time");
        check(s.first == 1040 && (int) s.fed.size() == 40, "read 40 beside a partner: same tokens, same first token");
        s = simulate(4990, 4950, true, false);
        check(s.windows == 6 && s.first == 1000 + 4990 && s.fed.size() == 4990 && s.fed[4950] == 1000 + 4950,
              "resume 4950 of 4990: 40 rows from the held prefix on");
        s = simulate(40, 39, true, true);
        check(s.windows == 1 && s.first == 1040 && s.rows_max == 1, "only the last token left: one row, its pick");
        s = simulate(97, 0, true, true, 6);
        check(s.windows == 20 && s.rows_max == 5 && s.rows_ok && s.first == 1097, "97 tokens with 6-row hand-off: 20 windows of 5");
    }

    {   // the drafter's catch-up tokens
        const int32_t ids[6] = {10, 11, 12, 13, 14, 15};
        int32_t out[8] = {};
        fl::catchup_next(ids, 6, 0, 3, 99, out);
        check(out[0] == 11 && out[1] == 12 && out[2] == 13, "catch-up: row t pairs with the prompt token at p + t + 1");
        fl::catchup_next(ids, 6, 3, 3, 99, out);
        check(out[0] == 14 && out[1] == 15 && out[2] == 99, "catch-up: the prompt's last row pairs with the first token");
        const fl::Step a = fl::step(3, 0, 6), b = fl::step(3, 3, 6);
        check(a.keep == 3 && a.p == 3 && !a.done && a.pick_row == -1, "step: a middle window keeps all, emits nothing");
        check(b.keep == 3 && b.p == 6 && b.done && b.pick_row == 2, "step: the last window: its last row's pick is the first token");
    }

    std::printf("%s\n", g_fail == 0 ? "fast_lane_test: all passed" : "fast_lane_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
