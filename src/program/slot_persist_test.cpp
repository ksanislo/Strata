// slot_persist.hpp: which --batch slot conversation --session-dir saves before the slot is overwritten.
#include "strata/program/slot_persist.hpp"

#include <cstdio>

using strata::program::slot_ids_continue;
using strata::program::slot_persist_wanted;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    const std::vector<int32_t> a{1, 2, 3, 4}, a_more{1, 2, 3, 4, 5, 6}, a_turn{1, 2, 3, 9, 9}, b{7, 8, 9}, none;
    CHECK(slot_ids_continue(a_more, a));
    CHECK(slot_ids_continue(a, a));
    CHECK(!slot_ids_continue(a, a_more));
    CHECK(!slot_ids_continue(a_turn, a));
    CHECK(slot_ids_continue(a, none));

    // another conversation takes the slot: saved
    CHECK(slot_persist_wanted(true, false, a, b, {}));
    // not kept (running, tainted, pictures) or empty: nothing to save
    CHECK(!slot_persist_wanted(false, false, a, b, {}));
    CHECK(!slot_persist_wanted(true, false, none, b, {}));
    // the request continues this slot (its turn checkpoint: the next turn without the reply's thinking)
    CHECK(!slot_persist_wanted(true, true, a, a_turn, {}));
    // ... or continues what it holds, from elsewhere (a longer copy parked in RAM or on disk)
    CHECK(!slot_persist_wanted(true, false, a, a_more, {}));
    // a rewrite of the tail that did not come from the slot: the old tail is saved
    CHECK(slot_persist_wanted(true, false, a, a_turn, {}));
    // another slot holds it or more: saved from there
    CHECK(!slot_persist_wanted(true, false, a, b, {&b, &a_more}));
    CHECK(!slot_persist_wanted(true, false, a, b, {&a}));
    CHECK(slot_persist_wanted(true, false, a, b, {&b, &a_turn, nullptr}));
    // shutdown: nothing comes next
    CHECK(slot_persist_wanted(true, false, a, none, {}));

    if (fails == 0) std::printf("slot_persist_test: ok\n");
    return fails == 0 ? 0 : 1;
}
