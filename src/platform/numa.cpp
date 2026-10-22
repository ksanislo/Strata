// src/platform/numa.cpp - see include/strata/platform/numa.hpp.
#include "strata/platform/numa.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#if defined(__linux__)
#include <cerrno>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::platform {

int numa_node_count() {
#if defined(__linux__)
    static const int n = [] {
        std::FILE* f = std::fopen("/sys/devices/system/node/online", "r");
        if (f == nullptr) return 1;
        char line[256] = {0};
        const bool got = std::fgets(line, sizeof line, f) != nullptr;
        std::fclose(f);
        if (!got) return 1;
        int hi = 0;
        for (const char* s = line; *s;) {   // "0-1", "0,2-3": the last number is the highest
            if (std::isdigit((unsigned char) *s)) {
                char* e = nullptr;
                hi = std::max(hi, (int) std::strtol(s, &e, 10));
                s = e;
            } else {
                ++s;
            }
        }
        return std::min(hi + 1, 64);
    }();
    return n;
#else
    return 1;
#endif
}

int numa_node_of_pci(const std::string& bus_id) {
#if defined(__linux__)
    std::string id;
    for (const char c : bus_id) id += (char) std::tolower((unsigned char) c);
    const size_t colon = id.find(':');
    if (colon != std::string::npos && colon > 4) id = id.substr(colon - 4);   // CUDA's 8-digit domain -> sysfs' 4
    std::FILE* f = std::fopen(("/sys/bus/pci/devices/" + id + "/numa_node").c_str(), "r");
    if (f == nullptr) return -1;
    int node = -1;
    if (std::fscanf(f, "%d", &node) != 1) node = -1;
    std::fclose(f);
    return node;
#else
    (void) bus_id;
    return -1;
#endif
}

std::vector<int> numa_layer_nodes(int64_t n_layers, const std::vector<int64_t>& split_at,
                                  const std::vector<int>& stage_nodes) {
    std::vector<int> out((size_t) std::max<int64_t>(n_layers, 0), -1);
    if (stage_nodes.size() != split_at.size() + 1) return out;
    size_t k = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        while (k < split_at.size() && l >= split_at[k]) ++k;
        out[(size_t) l] = stage_nodes[k];
    }
    return out;
}

std::vector<NumaSpan> numa_spans(const std::vector<uint64_t>& layer_off, const std::vector<int>& layer_nodes,
                                 uint64_t total, uint64_t align) {
    std::vector<NumaSpan> out;
    const size_t n = std::min(layer_off.size(), layer_nodes.size());
    if (align == 0) align = 1;
    for (size_t l = 0; l < n; ++l) {
        uint64_t a = l == 0 ? 0 : layer_off[l] / align * align;
        if (!out.empty() && out.back().node == layer_nodes[l]) continue;
        if (!out.empty()) {
            a = std::max(a, out.back().off);
            out.back().len = a - out.back().off;
            if (out.back().len == 0) out.pop_back();
        }
        out.push_back({a, 0, layer_nodes[l]});
    }
    if (!out.empty()) out.back().len = total > out.back().off ? total - out.back().off : 0;
    out.erase(std::remove_if(out.begin(), out.end(), [](const NumaSpan& s) { return s.node < 0 || s.len == 0; }),
              out.end());
    return out;
}

int numa_place_mode() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_NUMA_PLACE");
        if (v == nullptr || v[0] == '\0' || v[0] == '0') return 0;
        return std::strcmp(v, "bind") == 0 ? 2 : 1;
    }();
    return m;
}

#if defined(__linux__)
namespace {
long page_size() {
    static const long p = sysconf(_SC_PAGESIZE) > 0 ? sysconf(_SC_PAGESIZE) : 4096;
    return p;
}
std::mutex g_mu;
std::map<void*, uint64_t>& owned() {
    static std::map<void*, uint64_t> m;
    return m;
}
}  // namespace

bool numa_bind(void* p, uint64_t bytes, int node, bool strict, std::string* err) {
    if (p == nullptr || bytes == 0 || node < 0 || node >= 64) {
        if (err) *err = "nothing to bind";
        return false;
    }
    unsigned long mask = 1ul << node;
    const long r = syscall(SYS_mbind, p, (unsigned long) bytes, strict ? MPOL_BIND : MPOL_PREFERRED, &mask,
                           (unsigned long) (8 * sizeof mask + 1), 0ul);
    if (r != 0 && err) *err = std::string("mbind: ") + std::strerror(errno);
    return r == 0;
}

std::vector<uint64_t> numa_resident_by_node(const void* p, uint64_t bytes, uint64_t step) {
    const int nn = numa_node_count();
    std::vector<uint64_t> out((size_t) nn + 1, 0);
    if (p == nullptr || bytes == 0) return out;
    const uint64_t ps = (uint64_t) page_size();
    step = std::max<uint64_t>(step / ps * ps, ps);
    const uintptr_t b = (uintptr_t) p / ps * ps, e = (uintptr_t) p + bytes;
    std::vector<void*> pages;
    std::vector<int> status;
    auto flush = [&]() {
        if (pages.empty()) return;
        status.assign(pages.size(), -1);
        if (syscall(SYS_move_pages, 0, (unsigned long) pages.size(), pages.data(), nullptr, status.data(), 0) != 0)
            std::fill(status.begin(), status.end(), -1);
        for (size_t i = 0; i < pages.size(); ++i) {
            const uintptr_t at = (uintptr_t) pages[i];
            const uint64_t span = std::min<uint64_t>(step, (uint64_t) (e - at));
            out[status[i] >= 0 && status[i] < nn ? (size_t) status[i] : (size_t) nn] += span;
        }
        pages.clear();
    };
    for (uintptr_t a = b; a < e; a += step) {
        pages.push_back((void*) a);
        if (pages.size() == 4096) flush();
    }
    flush();
    return out;
}

void* numa_map(uint64_t bytes, int node, bool strict) {
    if (bytes == 0 || node < 0) return nullptr;
    const uint64_t ps = (uint64_t) page_size();
    const uint64_t n = (bytes + ps - 1) / ps * ps;
    void* m = mmap(nullptr, (size_t) n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    if (!numa_bind(m, n, node, strict)) {
        munmap(m, (size_t) n);
        return nullptr;
    }
    std::lock_guard<std::mutex> g(g_mu);
    owned()[m] = n;
    return m;
}

bool numa_unmap(void* p) {
    uint64_t n = 0;
    {
        std::lock_guard<std::mutex> g(g_mu);
        auto it = owned().find(p);
        if (it == owned().end()) return false;
        n = it->second;
        owned().erase(it);
    }
    munmap(p, (size_t) n);
    return true;
}

bool numa_mapped(const void* p) {
    std::lock_guard<std::mutex> g(g_mu);
    return owned().count(const_cast<void*>(p)) != 0;
}
#else
bool numa_bind(void*, uint64_t, int, bool, std::string* err) {
    if (err) *err = "NUMA placement is Linux only";
    return false;
}
std::vector<uint64_t> numa_resident_by_node(const void*, uint64_t, uint64_t) { return std::vector<uint64_t>(2, 0); }
void* numa_map(uint64_t, int, bool) { return nullptr; }
bool numa_unmap(void*) { return false; }
bool numa_mapped(const void*) { return false; }
#endif

}  // namespace strata::platform
