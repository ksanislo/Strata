#pragma once

#include <cstdint>
#include <cstdlib>

// STRATA_SHORT_READS=1 (opt-in): a prompt read of a few hundred new tokens on a layer split without its fixed costs.
// Header-only rules, tested on the CPU (tests/program/short_reads_test.cpp).
//
//  1. The CPU prefill share on a layer split: a staged chunk hands the idle CPU pool the non-resident experts few of
//     its tokens route to (prefill.cpp, STRATA_PREFILL_CPU_SHARE), instead of every stage streaming each routed
//     expert it does not hold over its PCIe link, one stage after another.  Chunks below `share_max` are staged.
//  2. The lazy refill: the slots the prompt path borrowed are copied back on a stream of their own while the reply
//     decodes; an expert counts as resident again only once its copy has landed (until then the pool computes it).
namespace strata::program::short_reads {

inline bool env_on(const char* v) { return v != nullptr && v[0] != 0 && v[0] != '0'; }

/// The staged-chunk limit the split's share takes: STRATA_PREFILL_CPU_SHARE_MAX when set (at least 1024), else 3072.
inline int64_t share_max(const char* env_max) {
    if (env_max == nullptr) return 3072;
    const int64_t v = std::atoll(env_max);
    return v < 1024 ? 1024 : v;
}

/// Whether STRATA_SHORT_READS arms the share by default on a layer split (one GPU has its own default rule).
///   explicit_set: STRATA_PREFILL_CPU_SHARE is set (Prefill honours it whatever this says)
inline bool split_share_default(bool on, bool multi_gpu, bool share_pool, bool explicit_set, bool has_work, bool memory_ok) {
    return on && multi_gpu && share_pool && !explicit_set && has_work && memory_ok;
}

/// What keeps the lazy refill off (nullptr: it may run).  Each of these reads or moves the residency table or the
/// cache slots between windows on its own schedule, so the lent slots are refilled before the reply there as before.
struct LazySetup {
    bool batch = false;        ///< --batch slots: admissions and groups reuse the main session's loan
    bool pipe = false;         ///< --pipeline-windows: windows in flight across the poll
    bool kv_grow = false;      ///< --kv-grow / --vram-elastic: the cache tail is taken for K/V
    bool peer = false;         ///< a peer card's cache
    bool helpers = false;      ///< helper / remote expert caches
    bool head_alias = false;   ///< --prefill-head-alias: the loan is the output head
    bool adapt_async = false;  ///< --adapt-async: the tier's round runs beside the windows
    bool evict_sync = false;   ///< the tier's lagging evictions patch the device tables per launch
    bool ram_adapt = false;    ///< STRATA_RAM_ADAPT: RAM residency follows host_res between windows
    bool foresight = false;    ///< the foresight swap moves experts between windows
};
inline const char* lazy_off(const LazySetup& s) {
    if (s.batch) return "--batch";
    if (s.pipe) return "--pipeline-windows";
    if (s.kv_grow) return "the elastic K/V or VRAM";
    if (s.peer) return "a peer card";
    if (s.helpers) return "helper expert caches";
    if (s.head_alias) return "--prefill-head-alias";
    if (s.adapt_async) return "--adapt-async";
    if (s.evict_sync) return "the lagging adaptive tier";
    if (s.ram_adapt) return "STRATA_RAM_ADAPT";
    if (s.foresight) return "the foresight swap";
    return nullptr;
}

/// The part of a request's prompt time no named step accounts for (never below 0).
inline double rest_ms(double prompt_ms, double before_ms, double lend_ms, double read_ms, double windows_ms,
                      double ckpt_ms, double refill_ms) {
    const double r = prompt_ms - before_ms - lend_ms - read_ms - windows_ms - ckpt_ms - refill_ms;
    return r > 0.0 ? r : 0.0;
}

}  // namespace strata::program::short_reads
