#include "strata/core/vmm.hpp"

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>

namespace strata::core {
namespace {

struct Api {
    bool ok = false;
    decltype(&cuDeviceGetAttribute) attr = nullptr;
    decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};

template <class F> bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else   // the versioned query arrived in CUDA 12.5
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = (F) p;
    return true;
}

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api() {
    static Api& a = *new Api;
    static std::once_flag once;
    std::call_once(once, [] {
        a.ok = resolve("cuDeviceGetAttribute", a.attr) && resolve("cuMemGetAllocationGranularity", a.granularity) &&
               resolve("cuMemAddressReserve", a.reserve) && resolve("cuMemAddressFree", a.free_va) &&
               resolve("cuMemCreate", a.create) && resolve("cuMemRelease", a.release) && resolve("cuMemMap", a.map) &&
               resolve("cuMemUnmap", a.unmap) && resolve("cuMemSetAccess", a.access);
    });
    return a;
}

// per device (a layer split's stages each map their own): VMM support and the chunk size, looked up once
constexpr int kMaxDev = 64;
struct Dev {
    std::once_flag once;
    bool ok = false;
    uint64_t gran = 0;
};
const Dev* dev_info(int dev) {
    static Dev* d = new Dev[kMaxDev];
    if (dev < 0 && cudaGetDevice(&dev) != cudaSuccess) return nullptr;
    if (dev < 0 || dev >= kMaxDev) return nullptr;
    Dev& x = d[dev];
    std::call_once(x.once, [&x, dev] {
        const Api& a = api();
        if (!a.ok) return;
        int vmm = 0;
        if (a.attr(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) dev) != CUDA_SUCCESS || !vmm)
            return;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev;
        size_t g = 0;
        if (a.granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || g == 0) return;
        x.gran = g;
        x.ok = true;
    });
    return x.ok ? &x : nullptr;
}
int current_dev() {
    int d = -1;
    return cudaGetDevice(&d) == cudaSuccess ? d : -1;
}

}  // namespace

bool vmm_available(int dev) { return dev_info(dev) != nullptr; }
uint64_t vmm_granularity(int dev) {
    const Dev* d = dev_info(dev);
    return d ? d->gran : 0;
}

VmmChunk vmm_chunk_new(int dev) {
    if (dev < 0) dev = current_dev();
    const Dev* d = dev_info(dev);
    if (d == nullptr) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    CUmemGenericAllocationHandle h = 0;
    if (api().create(&h, (size_t) d->gran, &prop, 0) != CUDA_SUCCESS) return 0;
    return (VmmChunk) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) api().release((CUmemGenericAllocationHandle) h);
}

bool VmmRange::reserve(uint64_t bytes, int dev) {
    release();
    if (dev < 0) dev = current_dev();
    const Dev* d = dev_info(dev);
    if (d == nullptr || bytes == 0) return false;
    const uint64_t n = (bytes + d->gran - 1) / d->gran;
    CUdeviceptr p = 0;
    if (api().reserve(&p, (size_t) (n * d->gran), 0, 0, 0) != CUDA_SUCCESS) return false;
    base_ = (unsigned long long) p;
    gran_ = d->gran;
    dev_ = dev;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    api().free_va((CUdeviceptr) base_, (size_t) ((uint64_t) h_.size() * gran_));
    base_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    if (api().map((CUdeviceptr) (base_ + (uint64_t) i * gran_), (size_t) gran_, 0, (CUmemGenericAllocationHandle) h, 0) !=
        CUDA_SUCCESS)
        return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = dev_;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return api().access((CUdeviceptr) (base_ + (uint64_t) lo * gran_), (size_t) ((uint64_t) (hi - lo) * gran_), &d, 1) ==
           CUDA_SUCCESS;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const VmmChunk h = h_[(size_t) i];
    if (api().unmap((CUdeviceptr) (base_ + (uint64_t) i * gran_), (size_t) gran_) != CUDA_SUCCESS) return 0;
    h_[(size_t) i] = 0;
    return h;
}

}  // namespace strata::core

#else  // not CUDA
// HIP (ROCm 6+): the same elastic range through hipMemAddressReserve / hipMemCreate / hipMemMap / hipMemSetAccess.  Available
// only where the device reports virtual memory management support (STRATA_HIP_VMM=0 turns it off): --kv-grow stays off
// otherwise, as before.  Everything else (Metal, gfx906, ROCm older than 6): no virtual memory, the elastic K/V stays off.
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#endif

#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 6
namespace strata::core {
namespace {
struct Api {
    bool ok = false;
    uint64_t gran = 0;
    int dev = -1;
};
void fail(const char* what, hipError_t e) {
    static int n = 0;
    if (n++ < 8) std::fprintf(stderr, "strata: HIP virtual memory: %s failed: %s\n", what, hipGetErrorString(e));
    (void) hipGetLastError();
}
hipMemAllocationProp alloc_prop(int dev) {
    hipMemAllocationProp p{};
    p.type = hipMemAllocationTypePinned;
    p.requestedHandleType = hipMemHandleTypeNone;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = dev;
    return p;
}
const Api& api() {
    static Api& a = *new Api;   // never destroyed: the K/V pools and the cache release their chunks from static destructors
    static std::once_flag once;
    std::call_once(once, [] {
        const char* off = std::getenv("STRATA_HIP_VMM");
        if (off != nullptr && off[0] == '0') return;
        if (hipGetDevice(&a.dev) != hipSuccess) { (void) hipGetLastError(); return; }
        int vmm = 0;
        if (hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, a.dev) != hipSuccess || !vmm) {
            (void) hipGetLastError();
            return;
        }
        const hipMemAllocationProp prop = alloc_prop(a.dev);
        size_t g = 0;
        if (hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum) != hipSuccess || g == 0) {
            (void) hipGetLastError();
            return;
        }
        // gfx1201 reports a 4 KiB minimum granularity; the elastic range works in 2 MiB chunks like CUDA's (any multiple of the minimum)
        const uint64_t want = 2ull << 20;
        a.gran = g >= want ? g : (want + g - 1) / g * g;
        a.ok = true;
    });
    return a;
}
}  // namespace

bool vmm_available(int) { return api().ok; }
uint64_t vmm_granularity(int) { return api().ok ? api().gran : 0; }

VmmChunk vmm_chunk_new(int) {
    const Api& a = api();
    if (!a.ok) return 0;
    const hipMemAllocationProp prop = alloc_prop(a.dev);
    hipMemGenericAllocationHandle_t h = nullptr;
    if (const hipError_t e = hipMemCreate(&h, (size_t) a.gran, &prop, 0); e != hipSuccess) { fail("hipMemCreate", e); return 0; }
    return (VmmChunk) (uintptr_t) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) (void) hipMemRelease((hipMemGenericAllocationHandle_t) (uintptr_t) h);
}

bool VmmRange::reserve(uint64_t bytes, int) {   // (one device under HIP)
    release();
    const Api& a = api();
    if (!a.ok || bytes == 0) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    gran_ = a.gran;
    dev_ = a.dev;
    void* p = nullptr;
    if (const hipError_t e = hipMemAddressReserve(&p, (size_t) (n * a.gran), 0, nullptr, 0); e != hipSuccess) { fail("hipMemAddressReserve", e); return false; }
    base_ = (unsigned long long) (uintptr_t) p;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    (void) hipMemAddressFree((void*) (uintptr_t) base_, (size_t) ((uint64_t) h_.size() * a.gran));
    base_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (const hipError_t e = hipMemMap((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0,
                                      (hipMemGenericAllocationHandle_t) (uintptr_t) h, 0);
        e != hipSuccess) {
        fail("hipMemMap", e);
        return false;
    }
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = a.dev;
    d.flags = hipMemAccessFlagsProtReadWrite;
    const hipError_t e = hipMemSetAccess((void*) (uintptr_t) (base_ + (uint64_t) lo * a.gran), (size_t) ((uint64_t) (hi - lo) * a.gran),
                                         &d, 1);
    const bool ok = e == hipSuccess;
    if (!ok) fail("hipMemSetAccess", e);
    return ok;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (hipMemUnmap((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), (size_t) a.gran) != hipSuccess) { (void) hipGetLastError(); return 0; }
    h_[(size_t) i] = 0;
    return h;
}
}  // namespace strata::core
#else


namespace strata::core {
bool vmm_available(int) { return false; }
uint64_t vmm_granularity(int) { return 0; }
VmmChunk vmm_chunk_new(int) { return 0; }
void vmm_chunk_free(VmmChunk) {}
bool VmmRange::reserve(uint64_t, int) { return false; }
void VmmRange::release() {}
int64_t VmmRange::mapped_count() const { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }
}  // namespace strata::core

#endif
#endif
