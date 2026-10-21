// src/platform/numa_test.cpp - STRATA_NUMA_PLACE's pure helpers, and (Linux, 2+ nodes) a bound mapping read back.
#include "strata/platform/numa.hpp"

#include <cstdio>
#include <cstring>

namespace {
int fails = 0;
#define CHECK(c)                                                                    \
    do {                                                                            \
        if (!(c)) {                                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);       \
            ++fails;                                                                \
        }                                                                           \
    } while (0)
}  // namespace

int main() {
    using namespace strata::platform;
    // the live split 11,24,36 on 4 cards: CUDA0/1 on node 0, CUDA2/3 on node 1
    {
        const auto n = numa_layer_nodes(48, {11, 24, 36}, {0, 0, 1, 1});
        CHECK(n.size() == 48);
        CHECK(n[0] == 0 && n[10] == 0 && n[11] == 0 && n[23] == 0);
        CHECK(n[24] == 1 && n[35] == 1 && n[36] == 1 && n[47] == 1);
        CHECK(numa_layer_nodes(48, {11, 24}, {0, 1}).front() == -1);   // wrong node count: nothing placed
    }
    // spans: equal layers merge; the node boundary is rounded down to the alignment
    {
        const uint64_t MiB = 1ull << 20, L = 2500 * MiB + 12345;
        std::vector<uint64_t> off;
        for (int l = 0; l < 48; ++l) off.push_back((uint64_t) l * L);
        const auto sp = numa_spans(off, numa_layer_nodes(48, {11, 24, 36}, {0, 0, 1, 1}), 48 * L + 7 * MiB, 2 * MiB);
        CHECK(sp.size() == 2);
        if (sp.size() == 2) {
            CHECK(sp[0].off == 0 && sp[0].node == 0);
            CHECK(sp[1].off == 24 * L / (2 * MiB) * (2 * MiB) && sp[1].node == 1);
            CHECK(sp[0].len == sp[1].off);
            CHECK(sp[1].off + sp[1].len == 48 * L + 7 * MiB);
        }
        // alternating nodes, and a layer smaller than the alignment that vanishes into its neighbour
        const auto alt = numa_spans({0, 4 * MiB, 5 * MiB, 8 * MiB}, {0, 1, 0, 1}, 10 * MiB, 2 * MiB);
        CHECK(alt.size() == 3);
        if (alt.size() == 3) {
            CHECK(alt[0].off == 0 && alt[0].len == 4 * MiB && alt[0].node == 0);
            CHECK(alt[1].off == 4 * MiB && alt[1].len == 4 * MiB && alt[1].node == 0);   // layer 1 rounds to nothing
            CHECK(alt[2].off == 8 * MiB && alt[2].len == 2 * MiB && alt[2].node == 1);
        }
        CHECK(numa_spans(off, std::vector<int>(48, -1), 48 * L, 2 * MiB).empty());
    }
    CHECK(numa_node_of_pci("ffff:ff:1f.7") == -1);
    // a bound mapping lands on its node (Linux, two nodes): what the arena's mbind relies on
    const int nn = numa_node_count();
    std::printf("numa_test: %d node(s)\n", nn);
#if defined(__linux__)
    for (int node = 0; node < nn && node < 2; ++node) {
        const uint64_t bytes = 64ull << 20;
        void* p = numa_map(bytes, node, false);
        CHECK(p != nullptr);
        if (p == nullptr) continue;
        CHECK(numa_mapped(p));
        auto before = numa_resident_by_node(p, bytes, 4096);
        CHECK(before.back() == bytes);   // untouched: nothing resident
        std::memset(p, 1, bytes);
        const auto got = numa_resident_by_node(p, bytes, 4096);
        std::printf("numa_test: mapping bound to node %d: %llu MiB there\n", node,
                    (unsigned long long) (got[(size_t) node] >> 20));
        CHECK(got[(size_t) node] == bytes);
        CHECK(numa_unmap(p));
        CHECK(!numa_unmap(p));
    }
#endif
    std::printf("numa_test: %s\n", fails == 0 ? "ok" : "FAILED");
    return fails == 0 ? 0 : 1;
}
