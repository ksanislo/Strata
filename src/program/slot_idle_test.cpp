// slot_idle.hpp: when a --batch slot's idle stretch (STRATA_ELASTIC_IDLE) starts over.
#include "strata/program/slot_idle.hpp"

#include <cstdio>

using strata::program::SlotIdleSeen;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    SlotIdleSeen<int> s;
    CHECK(!s.see(0, false, 0, 0));          // decoding: no stretch
    CHECK(!s.see(5, true, 1, 100));         // done and kept: the stretch starts at 5
    CHECK(s.since == 5);
    CHECK(s.see(9, true, 1, 100));          // the same conversation: it goes on from 5
    CHECK(s.since == 5);
    CHECK(!s.see(12, true, 2, 100));        // another request served in between (new stamp): starts over
    CHECK(s.since == 12);
    CHECK(!s.see(13, true, 2, 140));        // it grew (decoded on): starts over
    CHECK(s.since == 13);
    CHECK(!s.see(20, false, 2, 140));       // not kept (dropped, reading, queued): no stretch
    CHECK(!s.see(21, true, 2, 140));        // kept again: a new stretch from 21
    CHECK(s.since == 21 && s.see(30, true, 2, 140));
    if (fails == 0) std::printf("slot_idle_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
