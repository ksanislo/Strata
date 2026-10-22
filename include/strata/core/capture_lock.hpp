#pragma once
// A device-wide synchronize while any thread captures a graph invalidates that capture (stage threads capture batch
// windows while the main thread restores checkpoints): captures hold this lock shared, device syncs hold it exclusive.
#include <cuda_runtime.h>

#include <mutex>
#include <shared_mutex>

namespace strata::core {

inline std::shared_mutex& capture_mutex() {
    static std::shared_mutex m;
    return m;
}

inline std::shared_lock<std::shared_mutex>& capture_hold() {
    thread_local std::shared_lock<std::shared_mutex> hold;
    return hold;
}

// Held from begin_capture to end_capture only (instantiate and VRAM relief run after it); never device_sync() inside.
inline cudaError_t begin_capture(cudaStream_t s, cudaStreamCaptureMode mode) {
    auto& hold = capture_hold();
    if (!hold.owns_lock()) hold = std::shared_lock<std::shared_mutex>(capture_mutex());
    const cudaError_t e = cudaStreamBeginCapture(s, mode);
    if (e != cudaSuccess) hold.unlock();
    return e;
}

inline cudaError_t end_capture(cudaStream_t s, cudaGraph_t* graph) {
    const cudaError_t e = cudaStreamEndCapture(s, graph);
    auto& hold = capture_hold();
    if (hold.owns_lock()) hold.unlock();
    return e;
}

inline cudaError_t device_sync() {
    std::unique_lock<std::shared_mutex> lock(capture_mutex());
    return cudaDeviceSynchronize();
}

}  // namespace strata::core
