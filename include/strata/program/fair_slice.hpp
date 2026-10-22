#pragma once
// STRATA_FAIR_SLICE: time slices for the reads beside the decoding slots and the slots themselves.  Pure (fair_slice_test).

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::program {

struct FairSliceCfg {
    double cycle_s = 3.0;      ///< STRATA_FAIR_SLICE_S: the cycle every active request is served in, when pieces allow
    int64_t floor = 1024;      ///< STRATA_FAIR_SLICE_FLOOR: the smallest read piece (below it a piece costs as much)
    int64_t pmax = 2048;       ///< the read piece without slicing (STRATA_READ_YIELD_CHUNK, the loan, the prompt chunk)
    int quantum = 2;           ///< STRATA_FAIR_SLICE_QUANTUM: windows per decoding group in a decode turn, at least
    double rate0 = 300.0;      ///< read tokens/s of a turn (fill included) until one is measured
    double win0_s = 0.1;       ///< seconds per finished window during a decode turn until measured
    double cap_mult = 3.0;     ///< a decode turn ends after this many times its budget even short of its windows
};

/// The reads beside the slots go in read turns and the decoding slots in decode turns, one after the other.  Each of
/// the n = reads + decoding conversations gets about cycle_s / n per cycle: a read turn feeds pieces (one per read,
/// the read waiting longest first) sized to that slice, until the reads' share n_r * slice is spent (at least one
/// piece); a decode turn (all decoding slots advance together) lasts the decoders' share n_d * slice in windows.
/// One request alone (or only decoders) is not sliced at all.
class FairSlice {
public:
    enum class Phase { read, decode };

    explicit FairSlice(FairSliceCfg c = {}) : c_(c), rate_(c.rate0), win_s_(c.win0_s) {
        floor0_ = c_.floor;
        c_.floor = std::max<int64_t>(1, std::min(c_.floor, c_.pmax));
    }

    /// The normal piece of this run of reads (its loan and the prompt chunk can make it smaller).
    void set_pmax(int64_t p) {
        c_.pmax = std::max<int64_t>(1, p);
        c_.floor = std::max<int64_t>(1, std::min(floor0_, c_.pmax));
    }

    static bool shared(int n_r, int n_d) { return n_r > 0 && n_r + n_d > 1; }
    double slice(int n_r, int n_d) const { return c_.cycle_s / (double) std::max(1, n_r + n_d); }

    /// A read piece's tokens now: the slice's worth at the measured rate, within [floor, pmax] (multiples of 256 above
    /// the floor); pmax when not shared.
    int64_t piece(int n_r, int n_d) const {
        if (!shared(n_r, n_d)) return c_.pmax;
        int64_t p = (int64_t) (rate_ * slice(n_r, n_d));
        if (p > c_.floor) p = std::max(c_.floor, p / 256 * 256);
        return std::clamp(p, c_.floor, c_.pmax);
    }

    Phase phase() const { return ph_; }
    int64_t turn() const { return turn_; }
    bool feeding() const { return ph_ == Phase::read && !closed_; }

    /// Read turn: may another piece of `p` tokens go now?  The first piece of a turn always may.
    bool may_feed(int n_r, int n_d, int64_t p) const {
        if (!feeding()) return false;
        if (n_d == 0) return true;   // (reads only: round robin, no turns)
        if (fed_pieces_ == 0) return true;
        return (double) (fed_tokens_ + p) / rate_ <= (double) n_r * slice(n_r, n_d);
    }
    void fed(double now, int64_t tokens) {
        if (fed_pieces_ == 0) t0_ = now;
        ++fed_pieces_;
        fed_tokens_ += tokens;
    }
    /// No more pieces in this read turn (the budget is spent, or every read had its piece).
    void close() { if (ph_ == Phase::read && fed_pieces_ > 0) closed_ = true; }
    /// The read turn's pieces have all landed (nothing in flight): its rate is measured; with decoders a decode turn
    /// starts (`wins`: windows finished so far, `groups`: decoding groups), else the next read turn.
    void read_done(double now, int n_r, int n_d, int groups, int64_t wins) {
        if (ph_ != Phase::read || fed_pieces_ == 0) return;
        const double dt = now - t0_;
        if (closed_ && dt > 0.05) rate_ = rate_ * 0.5 + 0.5 * ((double) fed_tokens_ / dt);
        ++reads_n_;
        reads_s_ += dt;
        reads_pieces_ += fed_pieces_;
        fed_pieces_ = 0;
        fed_tokens_ = 0;
        closed_ = false;
        if (n_d > 0) {
            ph_ = Phase::decode;
            t0_ = now;
            w0_ = wins;
            budget_ = (double) n_d * slice(n_r, n_d);
            const int64_t q = (int64_t) c_.quantum * std::max(1, groups);
            target_ = std::max<int64_t>(q, (int64_t) std::llround(budget_ / win_s_));
            cap_ = std::max(c_.cap_mult * budget_, c_.cap_mult * (double) q * win_s_);
        } else {
            ++turn_;
        }
    }
    /// Decode turn: over when its windows are done, nobody decodes any more, or its time cap has passed.
    bool decode_over(double now, int n_d, int64_t wins) const {
        return ph_ == Phase::decode && (n_d == 0 || wins - w0_ >= target_ || now - t0_ >= cap_);
    }
    void decode_done(double now, int64_t wins) {
        if (ph_ != Phase::decode) return;
        const double dt = now - t0_;
        const int64_t w = wins - w0_;
        if (w > 0 && dt > 0.0) win_s_ = win_s_ * 0.5 + 0.5 * (dt / (double) w);
        ++decs_n_;
        decs_s_ += dt;
        decs_w_ += w;
        ph_ = Phase::read;
        ++turn_;
    }
    /// A new run of reads: a read turn first (the measured costs stay).
    void restart() {
        ph_ = Phase::read;
        fed_pieces_ = 0;
        fed_tokens_ = 0;
        closed_ = false;
        ++turn_;
    }

    double rate() const { return rate_; }
    double win_s() const { return win_s_; }
    int64_t target() const { return target_; }
    // totals for the run's line
    int64_t reads_n_ = 0, reads_pieces_ = 0, decs_n_ = 0, decs_w_ = 0;
    double reads_s_ = 0.0, decs_s_ = 0.0;

private:
    FairSliceCfg c_;
    int64_t floor0_ = 0;
    Phase ph_ = Phase::read;
    bool closed_ = false;
    int64_t turn_ = 0, fed_pieces_ = 0, fed_tokens_ = 0, w0_ = 0, target_ = 0;
    double t0_ = 0.0, budget_ = 0.0, cap_ = 0.0;
    double rate_, win_s_;
};

}  // namespace strata::program
