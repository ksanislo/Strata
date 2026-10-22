// fair_slice.hpp: piece sizes, read turns and decode turns (STRATA_FAIR_SLICE).
#include "strata/program/fair_slice.hpp"

#include <cstdio>

using strata::program::FairSlice;
using strata::program::FairSliceCfg;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    FairSliceCfg c;
    c.cycle_s = 3.0;
    c.floor = 1024;
    c.pmax = 2048;
    c.quantum = 2;
    c.rate0 = 1000.0;
    c.win0_s = 0.1;
    {   // alone (one read, nobody decoding): no slicing at all
        FairSlice f(c);
        CHECK(!FairSlice::shared(1, 0) && !FairSlice::shared(0, 4));
        CHECK(f.piece(1, 0) == 2048);
        CHECK(f.may_feed(1, 0, 2048));
    }
    {   // the slice shrinks with more requests, never below the floor; multiples of 256 above it
        FairSlice f(c);
        CHECK(f.piece(1, 1) == 1280);     // 1000 tok/s * 1.5 s = 1500, down to 256s
        CHECK(f.piece(2, 2) == 1024);     // 750 tokens -> the floor
        CHECK(f.piece(1, 7) == 1024);
        FairSliceCfg d = c;
        d.rate0 = 5000.0;
        FairSlice g(d);
        CHECK(g.piece(1, 1) == 2048);     // capped at the normal piece
    }
    {   // one read beside two decoders: one piece, then a decode turn of the decoders' share
        FairSlice f(c);
        CHECK(f.may_feed(1, 2, 1024));
        f.fed(0.0, 1024);
        CHECK(!f.may_feed(1, 2, 1024));   // 2048 tokens at 1000 tok/s > 1 s share
        f.close();
        CHECK(!f.feeding());
        f.read_done(2.0, 1, 2, 2, 100);   // 1024 tokens in 2 s: rate 1000 -> 756
        CHECK(f.phase() == FairSlice::Phase::decode);
        CHECK(f.rate() > 700.0 && f.rate() < 800.0);
        CHECK(f.target() == 20);          // 2 s share / 0.1 s per window
        CHECK(!f.decode_over(2.5, 2, 110));
        CHECK(f.decode_over(3.0, 2, 120));
        CHECK(f.decode_over(2.1, 0, 100));   // nobody decodes any more
        CHECK(f.decode_over(8.1, 2, 101));   // the cap (3 x 2 s)
        const int64_t t = f.turn();
        f.decode_done(4.0, 120);          // 20 windows in 2 s: 0.1 s each
        CHECK(f.phase() == FairSlice::Phase::read && f.turn() == t + 1);
        CHECK(f.win_s() > 0.099 && f.win_s() < 0.101);
        CHECK(f.decs_n_ == 1 && f.decs_w_ == 20 && f.reads_n_ == 1 && f.reads_pieces_ == 1);
    }
    {   // a decode turn has at least `quantum` windows per decoding group
        FairSlice f(c);
        f.fed(0.0, 1024);
        f.close();
        f.read_done(1.0, 3, 1, 1, 0);     // share 3/4 s
        CHECK(f.target() == 8);           // 0.75 / 0.1 -> 8 >= 2
        FairSlice g(c);
        g.fed(0.0, 1024);
        g.close();
        g.read_done(1.0, 7, 1, 4, 0);     // share 3/8 s: 4 windows, but 4 groups x 2
        CHECK(g.target() == 8);
    }
    {   // reads only: no turns, every piece may go (round robin)
        FairSlice f(c);
        f.fed(0.0, 1024);
        CHECK(f.may_feed(3, 0, 1024));
        f.read_done(1.0, 3, 0, 0, 0);
        CHECK(f.phase() == FairSlice::Phase::read);
    }
    {   // several reads: pieces while the reads' share lasts (3 reads + 1 decoder: 2.25 s at 1000 tok/s)
        FairSlice f(c);
        CHECK(f.piece(3, 1) == 1024);
        f.fed(0.0, 1024);
        CHECK(f.may_feed(3, 1, 1024));    // 2048 tokens = 2.05 s <= 2.25
        f.fed(0.1, 1024);
        CHECK(!f.may_feed(3, 1, 1024));
        f.restart();
        CHECK(f.feeding() && f.may_feed(3, 1, 1024));
    }
    {   // close() before any piece keeps the turn open (nothing was ready to read)
        FairSlice f(c);
        f.close();
        CHECK(f.feeding());
        f.read_done(1.0, 1, 1, 1, 0);
        CHECK(f.phase() == FairSlice::Phase::read);
    }
    {   // a run's smaller normal piece (its loan) caps the floor too
        FairSlice f(c);
        f.set_pmax(512);
        CHECK(f.piece(1, 0) == 512 && f.piece(3, 3) == 512);
        f.set_pmax(2048);
        CHECK(f.piece(3, 3) == 1024);
    }
    if (fails == 0) std::printf("fair_slice_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
