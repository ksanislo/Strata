// include/strata/platform/numa.hpp - host memory on the NUMA node of the card that uses it (STRATA_NUMA_PLACE=1).
//
// A layer split across sockets makes each socket a half-machine: its cards run their layers, and the expert RAM, the
// stage hand-offs and the KV copies of those layers are best on that socket's node. Raw syscalls (mbind, move_pages):
// no libnuma link. Everything is a no-op off Linux and on one node.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::platform {

/// Highest online node + 1 (1 when unknown or off Linux).
int numa_node_count();

/// The node a PCI device sits on ("0000:03:00.0", any case, a CUDA-style 8-digit domain accepted); -1 when unknown.
int numa_node_of_pci(const std::string& bus_id);

/// Per layer, the node of the stage that runs it: stage 0 runs [0, split_at[0]), stage k [split_at[k-1], split_at[k]),
/// the last stage to n_layers. `stage_nodes` has split_at.size() + 1 entries.
std::vector<int> numa_layer_nodes(int64_t n_layers, const std::vector<int64_t>& split_at,
                                  const std::vector<int>& stage_nodes);

struct NumaSpan {
    uint64_t off = 0, len = 0;
    int node = -1;
};
/// Merges consecutive layers on one node into spans of [0, total). `layer_off` holds each layer's start; a boundary
/// between two nodes is rounded down to `align` (huge pages stay whole; at most one page goes to the neighbour).
/// Layers with node < 0 get no span.
std::vector<NumaSpan> numa_spans(const std::vector<uint64_t>& layer_off, const std::vector<int>& layer_nodes,
                                 uint64_t total, uint64_t align);

/// STRATA_NUMA_PLACE: 0 off, 1 preferred (falls back to the other node when this one is full), 2 "bind" (strict).
int numa_place_mode();

/// mbind [p, p + bytes) to `node` before first touch (page-aligned start). False with `err` set on refusal.
bool numa_bind(void* p, uint64_t bytes, int node, bool strict, std::string* err = nullptr);

/// Resident bytes per node, sampling one page every `step` bytes (move_pages query); index numa_node_count() counts
/// the samples that are not resident.
std::vector<uint64_t> numa_resident_by_node(const void* p, uint64_t bytes, uint64_t step);

/// An untouched anonymous mapping bound to `node` (rounded up to whole pages); nullptr on failure. numa_unmap frees it.
void* numa_map(uint64_t bytes, int node, bool strict);
/// Unmaps a numa_map block; false (and nothing done) when `p` is not one.
bool numa_unmap(void* p);
/// Whether `p` is a numa_map block.
bool numa_mapped(const void* p);

}  // namespace strata::platform
