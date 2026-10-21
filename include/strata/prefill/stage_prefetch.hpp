#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

// STRATA_STAGE_PREFETCH=1 (opt-in): a staged prompt chunk (below stream_all_min()) copies each layer's hottest
// non-resident experts into a few extra prompt-path slots BEFORE the layer's routing is known - a later stage of a
// layer split from the moment the read starts (its link idles while the stages before it read), then one layer ahead
// as its own layers finish.  A prefetched expert the routing picks is read from its slot instead of being staged after
// the routing; one it does not pick only cost link time.  The same blobs, the same order of products: the same bits.
// Header-only rules, tested on the CPU (tests/prefill/stage_prefetch_test.cpp).
namespace strata::prefill::stage_prefetch {

inline bool env_on(const char* v) { return v != nullptr && v[0] != 0 && v[0] != '0'; }

/// The prefetch slots a staged layout adds (STRATA_STAGE_PREFETCH_SLOTS, default 64), at most `cap`; 0 when off.
inline int slots(bool on, const char* env_slots, int cap) {
    if (!on) return 0;
    const int v = env_slots != nullptr ? std::atoi(env_slots) : 64;
    return std::clamp(v, 0, std::max(cap, 0));
}

/// The experts prefetched per layer (STRATA_STAGE_PREFETCH_LAYER, default every slot: one layer ahead), 1..slots.
inline int per_layer(const char* env_layer, int n_slots) {
    if (n_slots <= 0) return 0;
    const int v = env_layer != nullptr ? std::atoi(env_layer) : n_slots;
    return std::clamp(v, 1, n_slots);
}

struct Entry {
    int32_t l, e;
};

/// The prefetch order of a stage's layers [lb, le): per layer, up to `per_layer` candidates (non-resident, DMA-able)
/// hottest first (`rank` lower = hotter, ties by id), layer after layer.  `starts` gets le - lb + 1 offsets.
template <class Candidate, class Rank>
std::vector<Entry> plan(int64_t lb, int64_t le, int32_t n_expert, int per_layer, Candidate&& candidate, Rank&& rank,
                        std::vector<size_t>& starts) {
    std::vector<Entry> out;
    starts.assign((size_t) std::max<int64_t>(le - lb, 0) + 1, 0);
    std::vector<std::pair<int64_t, int32_t>> c;
    for (int64_t l = lb; l < le; ++l) {
        starts[(size_t) (l - lb)] = out.size();
        c.clear();
        for (int32_t e = 0; e < n_expert; ++e)
            if (candidate(l, e)) c.emplace_back((int64_t) rank(l, e), e);
        const size_t k = std::min(c.size(), (size_t) std::max(per_layer, 0));
        std::partial_sort(c.begin(), c.begin() + (std::ptrdiff_t) k, c.end());
        for (size_t i = 0; i < k; ++i) out.push_back({(int32_t) l, c[i].second});
    }
    starts[(size_t) std::max<int64_t>(le - lb, 0)] = out.size();
    return out;
}

/// Entry i goes to slot i % n_slots; it may be issued once the entry that held that slot before it (i - n_slots)
/// was released (its layer's work recorded on the compute stream), i.e. while i < released + n_slots.
inline bool can_issue(size_t i, size_t released, int n_slots) { return n_slots > 0 && i < released + (size_t) n_slots; }

}  // namespace strata::prefill::stage_prefetch
