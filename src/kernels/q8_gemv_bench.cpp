// src/kernels/q8_gemv_bench.cpp - MODEL-UPDATE: the dense Q8_0 GEMVs of Unsloth's files (attention, SSM, shared expert,
// hyper-connections, head: all Q8_0 there) at the engine's shapes, 1..6 columns (a verify window's tokens), through
// native_mmvq - the call the engine makes - against the card's memory bandwidth.  Enough copies of each matrix are
// cycled to miss the GPU's last-level cache, as a layer's call does.
//
//     build/q8_gemv_bench [--cols 1,2,3,4,6] [--iters 50] [--bw GB/s] [--gguf a.gguf[,b.gguf]] [--ab]
//
// --gguf: the files' own Q8_0 matrices (n_out >= 64) by shape instead of random ones.  --ab: each shape and column count
// on the default kernel and on STRATA_SM75_Q8G's (A B B A), with both outputs compared bit for bit.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
struct Shape { const char* what; int n_in, n_out; };
const Shape SHAPES[] = {
    {"GDN qkv 2560->10240", 2560, 10240},  {"GDN gate 2560->6144", 2560, 6144}, {"GDN/QSA out 6144->2560", 6144, 2560},
    {"QSA q 2560->12288", 2560, 12288},    {"QSA k/v 2560->512", 2560, 512},    {"HC down 10240->320", 10240, 320},
    {"HC up 320->10240", 320, 10240},      {"shared gate/up 2560->640", 2560, 640}, {"shared down 640->2560", 640, 2560},
    {"head 2560->248320", 2560, 248320},
};
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}
std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(std::atoi(s.substr(from, c == std::string::npos ? std::string::npos : c - from).c_str()));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> cols = {1, 2, 3, 4, 6};
    int iters = 50;
    double bw = 0;
    bool ab = false;
    std::string ggufs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cols" && i + 1 < argc) cols = ints(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--bw" && i + 1 < argc) bw = std::atof(argv[++i]);
        else if (a == "--gguf" && i + 1 < argc) ggufs = argv[++i];
        else if (a == "--ab") ab = true;
    }
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("q8_gemv_bench: %s%s\n", prop.name, ab ? "  (A = default kernel, B = STRATA_SM75_Q8G)" : "");
    std::mt19937 rng(7);

    // the matrices of each shape: real ones from --gguf, else random weights at the engine's shapes
    struct Mat { std::string what; int n_in, n_out; std::vector<std::vector<uint8_t>> copies; };
    std::vector<Mat> mats;
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    if (!ggufs.empty()) {
        std::map<std::pair<int, int>, Mat> by;
        std::stringstream ss(ggufs);
        std::string path;
        while (std::getline(ss, path, ',')) {
            files.push_back(std::make_unique<strata::GgufFile>(path));
            for (const auto& t : files.back()->tensors()) {
                if (t.type != 8 || t.shape.size() != 2 || t.shape[1] < 64) continue;
                const int n_in = (int) t.shape[0], n_out = (int) t.shape[1];
                Mat& m = by[{n_in, n_out}];
                const size_t bytes = (size_t) n_out * (n_in / 32) * 34;
                if (m.copies.empty()) { m.what = t.name; m.n_in = n_in; m.n_out = n_out; }
                if (m.copies.size() >= 16 || (!m.copies.empty() && bytes * (m.copies.size() + 1) > ((size_t) 512 << 20))) continue;
                const uint8_t* src = files.back()->tensor_data(t);
                m.copies.emplace_back(src, src + bytes);
            }
        }
        for (auto& kv : by) {
            kv.second.what += " (" + std::to_string(kv.second.n_in) + "->" + std::to_string(kv.second.n_out) + ", " +
                              std::to_string(kv.second.copies.size()) + " tensors)";
            mats.push_back(std::move(kv.second));
        }
    } else {
        for (const Shape& s : SHAPES) {
            const size_t bytes = (size_t) s.n_out * (s.n_in / 32) * 34;
            std::vector<uint8_t> host(bytes);
            for (size_t b = 0; b < bytes; b += 34) {   // fp16 scale 0.01 (0x211f), then random int8 weights
                host[b] = 0x1f;
                host[b + 1] = 0x21;
                for (int k = 0; k < 32; ++k) host[b + 2 + k] = (uint8_t) (rng() & 0xff);
            }
            const int copies = (int) std::max<size_t>(1, std::min<size_t>(16, ((size_t) 512 << 20) / bytes));
            mats.push_back({s.what, s.n_in, s.n_out, std::vector<std::vector<uint8_t>>(copies, host)});
        }
    }

    int differing = 0;
    for (const Mat& s : mats) {
        const size_t bytes = (size_t) s.n_out * (s.n_in / 32) * 34;
        // cycle enough copies to miss the GPU's last-level cache, as a layer's call does
        const int copies = (int) std::max<size_t>(s.copies.size(), std::min<size_t>(16, ((size_t) 512 << 20) / bytes));
        std::vector<void*> w(copies);
        for (int i = 0; i < copies; ++i) {
            ck(cudaMalloc(&w[i], bytes), "malloc w");
            ck(cudaMemcpy(w[i], s.copies[i % s.copies.size()].data(), bytes, cudaMemcpyHostToDevice), "copy w");
        }
        const int maxc = *std::max_element(cols.begin(), cols.end());
        std::vector<float> xh((size_t) s.n_in * maxc);
        for (auto& v : xh) v = std::normal_distribution<float>(0.f, 1.f)(rng);
        float *x = nullptr, *y = nullptr, *y2 = nullptr;
        void* xq = nullptr;
        ck(cudaMalloc(&x, xh.size() * 4), "x");
        ck(cudaMemcpy(x, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x copy");
        ck(cudaMalloc(&xq, K::native_q8_1_bytes(s.n_in, maxc)), "xq");
        ck(cudaMalloc(&y, (size_t) s.n_out * maxc * 4), "y");
        ck(cudaMalloc(&y2, (size_t) s.n_out * maxc * 4), "y2");
        cudaStream_t st;
        ck(cudaStreamCreate(&st), "stream");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        std::printf("\n%s: %.2f MB, %d copies\n", s.what.c_str(), bytes / 1e6, copies);
        auto time = [&](int c) {   // us per call
            for (int i = 0; i < 3; ++i) K::native_mmvq(8, w[i % copies], xq, y, s.n_in, s.n_out, c, st);
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) K::native_mmvq(8, w[i % copies], xq, y, s.n_in, s.n_out, c, st);
            cudaEventRecord(e1, st);
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1000.0 * ms / iters;
        };
        auto gb = [&](double us) { return bytes / (us * 1e3); };
        auto pct = [&](double us) { return bw > 0 ? " = " + std::to_string((int) (100 * gb(us) / bw + 0.5)) + "%" : std::string(); };
        for (int c : cols) {
            K::native_quantize_q8_1(x, xq, s.n_in, c, st);
            if (!ab) {
                const double us = time(c);
                std::printf("  %d col%s: %8.1f us  %7.1f GB/s%s of bw\n", c, c == 1 ? " " : "s", us, gb(us),
                            bw > 0 ? pct(us).c_str() : "");
                continue;
            }
            double t[2] = {0, 0};
            for (int arm : {0, 1, 1, 0}) {
                K::native_q8_0_turing_tune(arm, 0, -1);
                t[arm] += time(c) / 2;
            }
            // the outputs of the first copy, bit for bit
            const size_t n = (size_t) s.n_out * c;
            K::native_q8_0_turing_tune(0, 0, -1);
            K::native_mmvq(8, w[0], xq, y, s.n_in, s.n_out, c, st);
            K::native_q8_0_turing_tune(1, 0, -1);
            const bool took = K::native_q8_0_turing_takes(s.n_in, s.n_out, c);
            cudaMemsetAsync(y2, 0xff, n * 4, st);
            K::native_mmvq(8, w[0], xq, y2, s.n_in, s.n_out, c, st);
            K::native_q8_0_turing_tune(-1, 0, -1);
            std::vector<uint32_t> a(n), b(n);
            ck(cudaMemcpy(a.data(), y, n * 4, cudaMemcpyDeviceToHost), "read a");
            ck(cudaMemcpy(b.data(), y2, n * 4, cudaMemcpyDeviceToHost), "read b");
            size_t diff = 0;
            for (size_t i = 0; i < n; ++i) diff += a[i] != b[i];
            differing += diff != 0;
            std::printf("  %d col%s: A %8.1f us %6.1f GB/s%-5s  B %8.1f us %6.1f GB/s%-5s  B/A %.3f  %s%s\n", c,
                        c == 1 ? " " : "s", t[0], gb(t[0]), pct(t[0]).c_str(), t[1], gb(t[1]), pct(t[1]).c_str(),
                        t[1] / t[0], took ? "" : "(B = default kernel) ",
                        diff ? ("DIFFER in " + std::to_string(diff) + " outputs").c_str() : "bitwise equal");
        }
        for (auto p : w) cudaFree(p);
        cudaFree(x);
        cudaFree(xq);
        cudaFree(y);
        cudaFree(y2);
        cudaStreamDestroy(st);
    }
    if (ab) std::printf("\nq8_gemv_bench: %d shape/column count(s) differing\n", differing);
    return differing ? 1 : 0;
}
