// short_reads.hpp: STRATA_SHORT_READS's rules (synthetic: no model, no GPU).
#include "strata/program/short_reads.hpp"

#include <cstdio>
#include <cstring>
#include <iterator>

namespace sr = strata::program::short_reads;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    CHECK(!sr::env_on(nullptr));
    CHECK(!sr::env_on(""));
    CHECK(!sr::env_on("0"));
    CHECK(sr::env_on("1"));
    CHECK(sr::env_on("yes"));

    // the staged-chunk limit: 3072 unless named, never below the stream-all floor
    CHECK(sr::share_max(nullptr) == 3072);
    CHECK(sr::share_max("2048") == 2048);
    CHECK(sr::share_max("512") == 1024);
    CHECK(sr::share_max("junk") == 1024);
    CHECK(sr::share_max("8192") == 8192);

    // the split's share: only with the switch, on a split, with the pool, not named explicitly, work and memory
    CHECK(sr::split_share_default(true, true, true, false, true, true));
    CHECK(!sr::split_share_default(false, true, true, false, true, true));   // switch off: unchanged
    CHECK(!sr::split_share_default(true, false, true, false, true, true));   // one GPU has its own rule
    CHECK(!sr::split_share_default(true, true, false, false, true, true));   // --batch / --no-pool: no idle pool
    CHECK(!sr::split_share_default(true, true, true, true, true, true));     // STRATA_PREFILL_CPU_SHARE decides
    CHECK(!sr::split_share_default(true, true, true, false, false, true));   // every expert resident
    CHECK(!sr::split_share_default(true, true, true, false, true, false));   // RAM is short

    // the lazy refill: runs with nothing set; each setting alone keeps it off with a reason of its own
    CHECK(sr::lazy_off({}) == nullptr);
    bool sr::LazySetup::*const each[] = {&sr::LazySetup::batch,      &sr::LazySetup::pipe,        &sr::LazySetup::kv_grow,
                                         &sr::LazySetup::peer,       &sr::LazySetup::helpers,     &sr::LazySetup::head_alias,
                                         &sr::LazySetup::adapt_async, &sr::LazySetup::evict_sync, &sr::LazySetup::ram_adapt,
                                         &sr::LazySetup::foresight};
    const char* named[std::size(each)] = {};
    for (size_t i = 0; i < std::size(each); ++i) {
        sr::LazySetup s;
        s.*each[i] = true;
        named[i] = sr::lazy_off(s);
        CHECK(named[i] != nullptr);
        for (size_t j = 0; j < i; ++j) CHECK(named[j] == nullptr || std::strcmp(named[i], named[j]) != 0);
    }

    // the breakdown's rest: what the steps leave, never negative
    CHECK(sr::rest_ms(1000, 100, 10, 600, 50, 40, 30) == 170);
    CHECK(sr::rest_ms(100, 50, 10, 60, 0, 0, 0) == 0);

    if (fails == 0) std::printf("short_reads_test: ok\n");
    return fails == 0 ? 0 : 1;
}
