// fair_slice.hpp: slices, read turns and decode turns (STRATA_FAIR_SLICE).
#include "strata/program/fair_slice.hpp"

#include <cstdio>

using strata::program::FairSlice;
using strata::program::FairSliceCfg;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    FairSliceCfg c;   // defaults: 3 s slices, 30 s cycle cap, 1 s minimum, floor = pmax = 2048, quantum 2
    {   // alone, only decoders, only reads: no turns, full pieces
        FairSlice f(c);
        CHECK(!FairSlice::shared(1, 0) && !FairSlice::shared(0, 4) && FairSlice::shared(2, 0));
        CHECK(f.piece(1, 0) == 2048 && f.piece(3, 5) == 2048);
        f.fed(0.0, 2048, 0);
        CHECK(f.may_feed(100.0, 3, 0) && !f.read_over(100.0, 3, 0));
    }
    {   // slices: 3 s each, shrinking past 10 requests to keep a 30 s cycle, never below 1 s
        FairSlice f(c);
        CHECK(f.slice(5, 3) == 3.0);
        CHECK(f.slice(10, 5) == 2.0);
        CHECK(f.slice(40, 40) == 1.0);
    }
    {   // 5 reads + 3 decoders: a 15 s read turn, then a 9 s decode turn from the close
        FairSlice f(c);
        CHECK(f.may_feed(0.0, 5, 3));
        f.fed(0.0, 2048, 3);
        CHECK(f.may_feed(14.9, 5, 3) && !f.read_over(14.9, 5, 3));
        f.fed(3.0, 2048, 3);
        CHECK(f.read_over(15.0, 5, 3) && !f.may_feed(15.0, 5, 3));
        f.close(15.0, 5, 3, 2, 100);
        CHECK(f.phase() == FairSlice::Phase::decode && f.budget() == 9.0);
        CHECK(!f.decode_over(30.0, 3, 300));     // not drained yet: the pieces in flight still land
        f.drained(18.0, 110);                     // 4096 tokens in 18 s
        CHECK(f.rate() > 400.0 && f.rate() < 420.0);
        CHECK(!f.decode_over(23.0, 3, 200));     // 8 s of 9
        CHECK(!f.decode_over(24.0, 3, 112));     // 9 s, but only 2 of 4 windows to itself
        CHECK(f.decode_over(24.0, 3, 114));
        CHECK(f.decode_over(18.5, 0, 110));      // nobody decodes any more
        CHECK(f.decode_over(45.0, 3, 110));      // the cap: 3 x 9 s after the drain
        f.decode_done(24.0, 170);                 // 60 windows in 6 s after the drain
        CHECK(f.phase() == FairSlice::Phase::read && f.feeding());
        CHECK(f.win_s() > 0.09 && f.win_s() < 0.11);
        CHECK(f.reads_n_ == 1 && f.reads_pieces_ == 2 && f.decs_n_ == 1 && f.decs_w_ == 70);
        CHECK(f.reads_s_ == 15.0 && f.decs_s_ == 9.0);
    }
    {   // reads alone for a while, then a decoder: the turn's clock started at the last piece without decoders
        FairSlice f(c);
        f.fed(0.0, 2048, 0);
        f.fed(20.0, 2048, 0);
        CHECK(f.turn_pieces() == 1);
        CHECK(!f.read_over(22.0, 1, 1));         // 1 read + 1 decoder: 3 s share from 20
        CHECK(f.read_over(23.0, 1, 1));
    }
    {   // close without decoders, or before any piece: still reading
        FairSlice f(c);
        f.close(1.0, 2, 1, 1, 0);
        CHECK(f.phase() == FairSlice::Phase::read);
        f.fed(1.0, 2048, 0);
        f.close(2.0, 2, 0, 0, 0);
        CHECK(f.phase() == FairSlice::Phase::read);
    }
    {   // a floor below pmax: the slice's worth at the measured rate, multiples of 256
        FairSliceCfg d = c;
        d.floor = 1024;
        d.rate0 = 500.0;
        FairSlice f(d);
        CHECK(f.piece(1, 0) == 2048);             // alone: full
        CHECK(f.piece(2, 2) == 1280);             // 500 tok/s x 3 s = 1500
        CHECK(f.piece(20, 20) == 1024);           // 1 s slices: 500 -> the floor
        f.set_pmax(512);
        CHECK(f.piece(2, 2) == 512 && f.piece(1, 0) == 512);
    }
    if (fails == 0) std::printf("fair_slice_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
