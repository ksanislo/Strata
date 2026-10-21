// src/core/numa_place.cpp - see include/strata/core/numa_place.hpp.
#include "strata/core/numa_place.hpp"

#include "strata/platform/numa.hpp"

#include <atomic>
#include <set>

namespace strata::core {

namespace {
constexpr int kMaxDev = 64;
std::atomic<bool> g_on{false};
int g_node[kMaxDev];
std::atomic<uint64_t> g_placed[64];
}  // namespace

int device_numa_node(int dev) {
    char bus[64] = {0};
    if (cudaDeviceGetPCIBusId(bus, (int) sizeof bus, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return -1;
    }
    return strata::platform::numa_node_of_pci(bus);
}

bool numa_place_configure(const std::vector<int>& devices, std::string& why) {
    g_on = false;
    if (strata::platform::numa_place_mode() == 0) { why = "off (STRATA_NUMA_PLACE unset)"; return false; }
    const int nn = strata::platform::numa_node_count();
    if (nn < 2) { why = "off: one NUMA node"; return false; }
    for (int& n : g_node) n = -1;
    std::set<int> nodes;
    why.clear();
    for (const int d : devices) {
        if (d < 0 || d >= kMaxDev) continue;
        g_node[d] = device_numa_node(d);
        if (g_node[d] >= 0) nodes.insert(g_node[d]);
        why += (why.empty() ? "" : ", ") + std::string("CUDA") + std::to_string(d) + " node " + std::to_string(g_node[d]);
    }
    if (nodes.size() < 2) { why = "off: the cards share one node (" + why + ")"; return false; }
    g_on = true;
    return true;
}

bool numa_place_on() { return g_on.load(); }

int numa_place_node(int dev) { return g_on && dev >= 0 && dev < kMaxDev ? g_node[dev] : -1; }

int numa_place_current_node() {
    if (!g_on) return -1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return -1;
    }
    return numa_place_node(dev);
}

cudaError_t numa_host_alloc(void** p, size_t bytes, unsigned flags, int node) {
    // (write-combined has no registration equivalent: those stay cudaHostAlloc)
    if (node >= 0 && node < 64 && (flags & cudaHostAllocWriteCombined) == 0) {
        if (void* m = strata::platform::numa_map(bytes, node, strata::platform::numa_place_mode() == 2)) {
            if (cudaHostRegister(m, bytes, flags & (cudaHostRegisterPortable | cudaHostRegisterMapped)) == cudaSuccess) {
                *p = m;
                g_placed[node] += bytes;
                return cudaSuccess;
            }
            (void) cudaGetLastError();
            strata::platform::numa_unmap(m);
        }
    }
    return cudaHostAlloc(p, bytes, flags);
}

void numa_host_free(void* p) {
    if (p == nullptr) return;
    if (strata::platform::numa_mapped(p)) {
        if (cudaHostUnregister(p) != cudaSuccess) (void) cudaGetLastError();
        strata::platform::numa_unmap(p);
        return;
    }
    (void) cudaFreeHost(p);
}

std::vector<uint64_t> numa_host_placed() {
    std::vector<uint64_t> out((size_t) strata::platform::numa_node_count(), 0);
    for (size_t i = 0; i < out.size() && i < 64; ++i) out[i] = g_placed[i].load();
    return out;
}

}  // namespace strata::core
