#pragma once
// Bulk copies that may cross separately pinned host ranges.  STRATA_KV_HOST_ELASTIC pins a slot's KV-streaming host
// copy in pieces of kSplitCopyBytes at addresses aligned to it, and cudaMemcpyAsync refuses ("invalid argument") a
// copy whose host side spans two registrations; split at the same boundaries of both sides, no piece does.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::kernels {

constexpr size_t kSplitCopyBytes = (size_t) 2 << 20;

inline cudaError_t memcpy_async_split(void* dst, const void* src, size_t n, cudaMemcpyKind kind, cudaStream_t stream) {
    auto* d = (uint8_t*) dst;
    auto* s = (const uint8_t*) src;
    while (n > 0) {
        size_t m = std::min(n, kSplitCopyBytes - (size_t) ((uintptr_t) s % kSplitCopyBytes));
        m = std::min(m, kSplitCopyBytes - (size_t) ((uintptr_t) d % kSplitCopyBytes));
        if (const cudaError_t e = cudaMemcpyAsync(d, s, m, kind, stream); e != cudaSuccess) return e;
        d += m;
        s += m;
        n -= m;
    }
    return cudaSuccess;
}

}  // namespace strata::kernels
