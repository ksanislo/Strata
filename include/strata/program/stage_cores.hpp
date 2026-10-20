// include/strata/program/stage_cores.hpp - STRATA_STAGE_SERVE: which CPU cores serve which stage of a layer split.
//
// Every stage gets a share of the physical cores of the NUMA node its GPU hangs off: the first core of the share runs
// the stage's service thread (its doorbells, and it drains with the pool), the rest are its expert pool's workers.
// The planning is pure (header-only, tested by stage_cores_test); the sysfs readers below it are Linux-only.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace strata::program {

struct StageCores {
    int node = -1;              ///< the node whose cores it got (-1: every core, the node was not known)
    int service = -1;           ///< the service thread's core (-1: no core was left)
    std::vector<int> workers;   ///< the pool's cores; {service} when the share was a single core
};

/// `total` units cut by `weight` (D'Hondt: the next unit to the largest weight / (units + 1)), each at most `cap`.
inline std::vector<int> split_by_weight(int total, const std::vector<int>& weight, const std::vector<int>& cap) {
    std::vector<int> out(weight.size(), 0);
    for (int u = 0; u < total; ++u) {
        int best = -1;
        for (size_t i = 0; i < weight.size(); ++i) {
            if (out[i] >= cap[i]) continue;
            // weight[i] / (out[i] + 1) > weight[best] / (out[best] + 1), ties to the first stage
            if (best < 0 || (long long) std::max(weight[i], 1) * (out[(size_t) best] + 1) >
                                (long long) std::max(weight[(size_t) best], 1) * (out[i] + 1))
                best = (int) i;
        }
        if (best < 0) break;
        ++out[(size_t) best];
    }
    return out;
}

/// The cores of each stage.  `stage_node[i]`: the node of stage i's GPU (-1: unknown); `weight[i]`: its layers.
/// `node_cores[n]`: node n's physical cores (one logical CPU each); `node_cores[-1]` lists every core and takes all
/// stages when any stage's node is unknown or not listed.  `exclude`: CPUs no stage gets (the host thread's core).
/// The stages of a node share its cores by weight, at least one each; `workers_total` > 0 caps the pool workers of
/// all stages together (split by weight; the service threads are not counted).
inline std::vector<StageCores> plan_stage_cores(const std::vector<int>& stage_node, const std::vector<int>& weight,
                                                const std::map<int, std::vector<int>>& node_cores,
                                                const std::vector<int>& exclude, int workers_total) {
    const size_t n = stage_node.size();
    std::vector<StageCores> out(n);
    bool known = true;
    for (int nd : stage_node) known = known && nd >= 0 && node_cores.count(nd) != 0;
    for (size_t i = 0; i < n; ++i) out[i].node = known ? stage_node[i] : -1;
    std::vector<int> nodes;
    for (const StageCores& s : out)
        if (std::find(nodes.begin(), nodes.end(), s.node) == nodes.end()) nodes.push_back(s.node);
    for (int nd : nodes) {
        std::vector<int> cores;
        if (auto it = node_cores.find(nd); it != node_cores.end())
            for (int c : it->second)
                if (std::find(exclude.begin(), exclude.end(), c) == exclude.end()) cores.push_back(c);
        std::vector<size_t> st;
        std::vector<int> w;
        for (size_t i = 0; i < n; ++i)
            if (out[i].node == nd) { st.push_back(i); w.push_back(i < weight.size() ? weight[i] : 1); }
        if (cores.empty()) continue;
        if (cores.size() < st.size()) {   // fewer cores than stages: they share, one core each
            for (size_t j = 0; j < st.size(); ++j) out[st[j]].service = cores[j % cores.size()];
            continue;
        }
        const std::vector<int> extra = split_by_weight((int) (cores.size() - st.size()), w, std::vector<int>(w.size(), (int) cores.size()));
        size_t at = 0;
        for (size_t j = 0; j < st.size(); ++j) {
            StageCores& s = out[st[j]];
            s.service = cores[at++];
            for (int k = 0; k < extra[j]; ++k) s.workers.push_back(cores[at++]);
        }
    }
    if (workers_total > 0) {
        std::vector<int> cap, w;
        int have = 0;
        for (size_t i = 0; i < n; ++i) {
            cap.push_back((int) out[i].workers.size());
            w.push_back(i < weight.size() ? weight[i] : 1);
            have += cap.back();
        }
        const std::vector<int> keep = split_by_weight(std::min(workers_total, have), w, cap);
        for (size_t i = 0; i < n; ++i) out[i].workers.resize((size_t) keep[i]);
    }
    for (StageCores& s : out)
        if (s.workers.empty() && s.service >= 0) s.workers.push_back(s.service);
    return out;
}

/// "0-9,20-29" -> {0..9, 20..29}; false (and `out` empty) on anything else.
inline bool parse_cpulist(const std::string& text, std::vector<int>& out) {
    out.clear();
    size_t p = 0;
    while (p < text.size() && text[p] != '\n') {
        size_t q = p;
        while (q < text.size() && text[q] >= '0' && text[q] <= '9') ++q;
        if (q == p) { out.clear(); return false; }
        const int lo = std::stoi(text.substr(p, q - p));
        int hi = lo;
        if (q < text.size() && text[q] == '-') {
            size_t r = ++q;
            while (r < text.size() && text[r] >= '0' && text[r] <= '9') ++r;
            if (r == q) { out.clear(); return false; }
            hi = std::stoi(text.substr(q, r - q));
            q = r;
        }
        if (hi < lo) { out.clear(); return false; }
        for (int c = lo; c <= hi; ++c) out.push_back(c);
        if (q < text.size() && text[q] == ',') ++q;
        p = q;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return !out.empty();
}

/// One line for the log: "CUDA2 node 1: service 10, workers 11,12,13,14".
inline std::string stage_cores_text(const StageCores& s) {
    std::string t = "node " + std::to_string(s.node) + ": service " + std::to_string(s.service) + ", workers ";
    for (size_t i = 0; i < s.workers.size(); ++i) t += (i ? "," : "") + std::to_string(s.workers[i]);
    return t;
}

#if defined(__linux__)
inline std::string read_sysfs_line(const std::string& path) {
    std::string s;
    if (std::FILE* f = std::fopen(path.c_str(), "r")) {
        char buf[4096];
        if (std::fgets(buf, sizeof buf, f)) s = buf;
        std::fclose(f);
    }
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

/// The NUMA node of a PCI device ("0000:3b:00.0", any case, or nvidia-smi's 8-digit domain), -1 when sysfs does not say.
inline int pci_numa_node(std::string bus_id) {
    for (char& c : bus_id) c = (char) std::tolower((unsigned char) c);
    if (const size_t colon = bus_id.find(':'); colon != std::string::npos && colon > 4) bus_id.erase(0, colon - 4);
    const std::string s = read_sysfs_line("/sys/bus/pci/devices/" + bus_id + "/numa_node");
    return s.empty() ? -1 : std::atoi(s.c_str());
}

/// Each NUMA node's physical cores among `allowed` (the first allowed logical CPU of each core), and all of them
/// under node -1.
inline std::map<int, std::vector<int>> numa_physical_cores(const std::vector<int>& allowed) {
    std::vector<int> firsts;   // one CPU per core, in `allowed` order
    for (int c : allowed) {
        std::vector<int> sib;
        if (!parse_cpulist(read_sysfs_line("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"), sib))
            sib = {c};
        bool first = true;
        for (int s : sib)
            if (s != c && std::find(allowed.begin(), allowed.end(), s) != allowed.end() && s < c) first = false;
        if (first) firsts.push_back(c);
    }
    std::map<int, std::vector<int>> out;
    out[-1] = firsts;
    for (int nd = 0; nd < 64; ++nd) {
        std::vector<int> list;
        if (!parse_cpulist(read_sysfs_line("/sys/devices/system/node/node" + std::to_string(nd) + "/cpulist"), list)) continue;
        std::vector<int>& v = out[nd];
        for (int c : firsts)
            if (std::binary_search(list.begin(), list.end(), c)) v.push_back(c);
    }
    return out;
}

/// The logical CPUs of `cpu`'s physical core ({cpu} when sysfs does not say).
inline std::vector<int> core_cpus(int cpu) {
    std::vector<int> sib;
    if (!parse_cpulist(read_sysfs_line("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list"), sib))
        sib = {cpu};
    return sib;
}
#endif

}  // namespace strata::program
