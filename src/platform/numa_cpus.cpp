// src/platform/numa_cpus.cpp - see numa_cpus.hpp.
#include "strata/platform/numa_cpus.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::platform {

std::vector<int> parse_cpulist(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty() || part.find_first_not_of("0123456789-\n ") != std::string::npos) continue;
        const size_t dash = part.find('-');
        char* end = nullptr;
        const long a = std::strtol(part.c_str(), &end, 10);
        if (end == part.c_str() || a < 0) continue;
        long b = a;
        if (dash != std::string::npos) {
            const char* q = part.c_str() + dash + 1;
            b = std::strtol(q, &end, 10);
            if (end == q || b < a) continue;
        }
        for (long c = a; c <= b && c < 65536; ++c) out.push_back((int) c);
    }
    return out;
}

std::vector<int> numa_node_cpus(int node) {
#if defined(__linux__)
    if (node < 0) return {};
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string line;
    if (!std::getline(f, line)) return {};
    return parse_cpulist(line);
#else
    (void) node;
    return {};
#endif
}

bool pin_this_thread(const std::vector<int>& cpus) {
#if defined(__linux__)
    if (cpus.empty()) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
        if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
#else
    (void) cpus;
    return false;
#endif
}

}  // namespace strata::platform
