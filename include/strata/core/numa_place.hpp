// include/strata/core/numa_place.hpp - STRATA_NUMA_PLACE=1: pinned host buffers on the NUMA node of their card.
//
// Off unless the switch is set, the box has more than one node and the configured devices span more than one; then
// every host_alloc_guarded buffer lands on the node of the CURRENT device (a stage's verifier, session and KV copy
// are allocated under its OnDevice), and numa_host_alloc places one on a node the caller names.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The node of CUDA device `dev` from its PCI address; -1 when unknown.
int device_numa_node(int dev);

/// Turns placement on for these devices (a layer split's, stage order) when STRATA_NUMA_PLACE is set and they span
/// more than one node. Returns whether it is on; `why` says what was decided.
bool numa_place_configure(const std::vector<int>& devices, std::string& why);
bool numa_place_on();
/// The node of `dev` while placement is on, else -1.
int numa_place_node(int dev);
/// The node of the current device while placement is on, else -1.
int numa_place_current_node();

/// cudaHostAlloc, or with node >= 0 an anonymous mapping bound to `node` and registered with the same
/// Portable/Mapped flags (cudaHostAlloc when that fails). Free with numa_host_free.
cudaError_t numa_host_alloc(void** p, size_t bytes, unsigned flags, int node);
/// Frees either kind.
void numa_host_free(void* p);

/// Bytes numa_host_alloc placed per node so far (for the startup line).
std::vector<uint64_t> numa_host_placed();

}  // namespace strata::core
