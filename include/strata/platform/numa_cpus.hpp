// include/strata/platform/numa_cpus.hpp - the CPUs of a NUMA node, and a thread pinned to them (STRATA_PARALLEL_LOAD:
// the thread that fills a layer's expert RAM runs on the node of the card that runs the layer).  Linux only; elsewhere
// a node has no CPUs listed and pinning does nothing.
#pragma once

#include <string>
#include <vector>

namespace strata::platform {

/// "0-9,20-29" -> {0..9, 20..29}; malformed parts are skipped.
std::vector<int> parse_cpulist(const std::string& s);
/// The online CPUs of `node` (/sys/devices/system/node/nodeN/cpulist); empty when unknown.
std::vector<int> numa_node_cpus(int node);
/// Pins the calling thread to `cpus`; false when empty or refused.
bool pin_this_thread(const std::vector<int>& cpus);

}  // namespace strata::platform
