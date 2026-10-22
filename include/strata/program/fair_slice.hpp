#pragma once
// STRATA_FAIR_SLICE: time slices for the reads beside the decoding slots and the slots themselves.  Pure (fair_slice_test).

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::program {

struct FairSliceCfg {
    double slice_s = 3.0;      ///< STRATA_FAIR_SLICE_S: each active request's slice of a cycle
    double cycle_s = 30.0;     ///< STRATA_FAIR_SLICE_CYCLE_S: with many requests the slices shrink to keep this cycle
    double min_slice_s = 1.0;  ///< ... but not below this
    int64_t floor = 2048;      ///< STRATA_FAIR_SLICE_FLOOR: the smallest read piece (default: never smaller than pmax)
    int64_t pmax = 2048;       ///< the read piece without slicing (STRATA_READ_YIELD_CHUNK, the loan, the prompt chunk)
    int quantum = 2;           ///< STRATA_FAIR_SLICE_QUANTUM: windows per decoding group a decode turn has to itself
    double rate0 = 600.0;      ///< read tokens/s until one is measured (only sizes pieces when floor < pmax)
    double win0_s = 0.1;       ///< seconds per finished window until measured (reported)
    double cap_mult = 3.0;     ///< a decode turn ends this many budgets after its pieces landed, windows or not
};

/// Beside decoding slots, read turns and decode turns alternate.  Each of the n active requests (reads with pieces
/// left + decoding conversations) gets a slice s = min(slice_s, max(min_slice_s, cycle_s / n)).  A read turn feeds
/// pieces (a read never fed first, then the one fed longest ago) for the reads' share n_r * s of wall time (at least
/// one piece); a decode turn starts when the read turn closes (the pieces stage 0 has not started are taken back by the
/// caller), lasts the decoders' share n_d * s - all decoding slots advance together in it - and has at least
/// `quantum` windows per decoding group after the last piece landed.  One request alone, or only decoders, or only
/// reads: no turns (reads only go round robin).
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
    double slice(int n_r, int n_d) const {
        const double n = (double) std::max(1, n_r + n_d);
        return std::min(c_.slice_s, std::max(c_.min_slice_s, c_.cycle_s / n));
    }
    /// A read piece's tokens now: pmax unless shared and a floor below it (then the slice's worth at the measured rate).
    int64_t piece(int n_r, int n_d) const {
        if (!shared(n_r, n_d) || c_.floor >= c_.pmax) return c_.pmax;
        int64_t p = (int64_t) (rate_ * slice(n_r, n_d));
        if (p > c_.floor) p = std::max(c_.floor, p / 256 * 256);
        return std::clamp(p, c_.floor, c_.pmax);
    }

    Phase phase() const { return ph_; }
    bool feeding() const { return ph_ == Phase::read; }
    int64_t turn_pieces() const { return fed_pieces_; }

    /// Read turn: may another piece go now?  Always with no decoder or as the turn's first piece.
    bool may_feed(double now, int n_r, int n_d) const {
        if (ph_ != Phase::read) return false;
        return n_d == 0 || fed_pieces_ == 0 || !read_over(now, n_r, n_d);
    }
    /// The reads' share of this turn is spent (decoders are waiting).
    bool read_over(double now, int n_r, int n_d) const {
        return ph_ == Phase::read && n_d > 0 && fed_pieces_ > 0 && now - t0_ >= (double) n_r * slice(n_r, n_d);
    }
    /// A piece fed.  With nobody decoding the turn's clock keeps restarting: a turn is the reads' time beside decoders.
    void fed(double now, int64_t tokens, int n_d) {
        if (fed_pieces_ == 0 || n_d == 0) {
            t0_ = now;
            fed_tokens_ = 0;
            fed_pieces_ = 0;
        }
        ++fed_pieces_;
        fed_tokens_ += tokens;
    }
    /// The read turn ends: with decoders the decode turn starts now (its windows run while the pieces in flight land).
    void close(double now, int n_r, int n_d, int groups, int64_t wins) {
        if (ph_ != Phase::read || fed_pieces_ == 0 || n_d == 0) return;
        ++reads_n_;
        reads_s_ += now - t0_;
        reads_pieces_ += fed_pieces_;
        ph_ = Phase::decode;
        tr0_ = t0_;
        t0_ = now;
        w0_ = wins;
        drained_ = false;
        budget_ = (double) n_d * slice(n_r, n_d);
        need_ = (int64_t) c_.quantum * std::max(1, groups);
    }
    /// Decode turn: the read turn's pieces have all landed (the cards are the windows' alone from here).
    void drained(double now, int64_t wins) {
        if (ph_ != Phase::decode || drained_) return;
        drained_ = true;
        td_ = now;
        wd_ = wins;
        if (now - tr0_ > 0.05) rate_ = rate_ * 0.5 + 0.5 * ((double) fed_tokens_ / (now - tr0_));
    }
    bool is_drained() const { return drained_; }
    /// Over when its share has passed and it had its quantum to itself, or nobody decodes, or its cap after the drain.
    bool decode_over(double now, int n_d, int64_t wins) const {
        if (ph_ != Phase::decode) return false;
        if (n_d == 0) return true;
        if (!drained_) return false;
        return (now - t0_ >= budget_ && wins - wd_ >= need_) || now - td_ >= c_.cap_mult * std::max(budget_, 1.0);
    }
    void decode_done(double now, int64_t wins) {
        if (ph_ != Phase::decode) return;
        if (drained_ && wins > wd_ && now > td_) win_s_ = win_s_ * 0.5 + 0.5 * ((now - td_) / (double) (wins - wd_));
        ++decs_n_;
        decs_s_ += now - t0_;
        decs_w_ += wins - w0_;
        ph_ = Phase::read;
        fed_pieces_ = 0;
        fed_tokens_ = 0;
    }
    /// A new run of reads: a read turn first (the measured costs stay).
    void restart() {
        ph_ = Phase::read;
        fed_pieces_ = 0;
        fed_tokens_ = 0;
    }

    double rate() const { return rate_; }
    double win_s() const { return win_s_; }
    double budget() const { return budget_; }
    // totals for the run's line
    int64_t reads_n_ = 0, reads_pieces_ = 0, decs_n_ = 0, decs_w_ = 0;
    double reads_s_ = 0.0, decs_s_ = 0.0;

private:
    FairSliceCfg c_;
    int64_t floor0_ = 0;
    Phase ph_ = Phase::read;
    bool drained_ = false;
    int64_t fed_pieces_ = 0, fed_tokens_ = 0, w0_ = 0, wd_ = 0, need_ = 0;
    double t0_ = 0.0, tr0_ = 0.0, td_ = 0.0, budget_ = 0.0;
    double rate_, win_s_;
};

}  // namespace strata::program
