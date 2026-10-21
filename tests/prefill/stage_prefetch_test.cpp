// stage_prefetch.hpp: STRATA_STAGE_PREFETCH's rules (synthetic: no model, no GPU).
#include "strata/prefill/stage_prefetch.hpp"

#include <cstdio>
#include <set>

namespace sp = strata::prefill::stage_prefetch;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    CHECK(!sp::env_on(nullptr));
    CHECK(!sp::env_on("0"));
    CHECK(sp::env_on("1"));

    // slots: 0 when off (the layout unchanged), 64 by default, never past the event arrays
    CHECK(sp::slots(false, "128", 480) == 0);
    CHECK(sp::slots(true, nullptr, 480) == 64);
    CHECK(sp::slots(true, "128", 480) == 128);
    CHECK(sp::slots(true, "4000", 480) == 480);
    CHECK(sp::slots(true, "-3", 480) == 0);
    CHECK(sp::per_layer(nullptr, 64) == 64);
    CHECK(sp::per_layer("16", 64) == 16);
    CHECK(sp::per_layer("100", 64) == 64);
    CHECK(sp::per_layer("0", 64) == 1);
    CHECK(sp::per_layer("8", 0) == 0);

    // the plan: per layer the hottest candidates (rank, then id), at most per_layer, layer after layer
    const int64_t lb = 11, le = 14;
    const int32_t ne = 8;
    auto cand = [](int64_t l, int32_t e) { return (e + l) % 3 != 0; };       // some resident
    auto rank = [](int64_t l, int32_t e) { return (int64_t) ((e * 5 + l) % 8); };
    std::vector<size_t> st;
    const auto pl = sp::plan(lb, le, ne, 3, cand, rank, st);
    CHECK(st.size() == 4);
    CHECK(st[0] == 0 && st[3] == pl.size());
    for (int64_t l = lb; l < le; ++l) {
        const size_t a = st[(size_t) (l - lb)], b = st[(size_t) (l - lb) + 1];
        CHECK(b - a == 3);
        std::set<int32_t> seen;
        for (size_t i = a; i < b; ++i) {
            CHECK(pl[i].l == l);
            CHECK(cand(l, pl[i].e));
            CHECK(seen.insert(pl[i].e).second);
            if (i > a) CHECK(rank(l, pl[i - 1].e) <= rank(l, pl[i].e));
        }
        // nothing hotter was left out
        int64_t worst = rank(l, pl[b - 1].e);
        for (int32_t e = 0; e < ne; ++e)
            if (cand(l, e) && !seen.count(e)) CHECK(rank(l, e) >= worst);
    }
    // a layer with fewer candidates than per_layer gives what it has; none at all, nothing
    auto few = [](int64_t l, int32_t e) { return l == 12 ? e == 5 : l == 13 ? false : true; };
    const auto pf = sp::plan(lb, le, ne, 4, few, rank, st);
    CHECK(st[1] - st[0] == 4 && st[2] - st[1] == 1 && st[3] - st[2] == 0);
    CHECK(pf[4].l == 12 && pf[4].e == 5);
    const auto pe = sp::plan(5, 5, ne, 4, few, rank, st);
    CHECK(pe.empty() && st.size() == 1);

    // the gate: the first n_slots entries go at once; entry i waits for entry i - n_slots's release
    CHECK(sp::can_issue(0, 0, 4) && sp::can_issue(3, 0, 4) && !sp::can_issue(4, 0, 4));
    CHECK(sp::can_issue(4, 1, 4) && !sp::can_issue(5, 1, 4));
    CHECK(!sp::can_issue(0, 0, 0));

    if (fails == 0) std::printf("stage_prefetch_test: ok\n");
    return fails == 0 ? 0 : 1;
}
