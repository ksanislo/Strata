// src/kernels/cpu/kq_avx1_parity.cpp - the AVX1 Q4_K / Q5_1 / Q8_0 expert rows (kq_avx1.cpp) against ggml-cpu's own
// per-token dots, bit for bit.  No GPU, no model: experts of the Flash-Next geometry (gate/up 640 x 2560, down
// 2560 x 640), quantized from random weights by ggml and also random bytes with fp16 scales of every kind
// (subnormal and negative included), activations quantized the way the engine quantizes them.
//
//     kq_avx1_parity            every row of every token count 1..9, 16, 64 and a few row ranges: memcmp-equal
//                               to ggml's vec_dot per token (what the engine runs with STRATA_CPU_KERNELS=old),
//                               directly and through native_gu_rows / native_down_rows.
//     kq_avx1_parity --bench [--pairs q4_K/q5_1,q8_0/q8_0] [--nt 1,2,4,8,64] [--threads 1,8,16]
//                    [--cpus 10-19,30-39] [--mb 1024] [--reps 5] [--part all|gu|down]
//                               whole experts (gate + up + down) per second, ggml's dot ("old") vs kq128 ("new") vs
//                               kq128 fast ("fast", STRATA_CPU_KERNELS=fast; Q8_0 has no fast form, so new = fast there),
//                               arms interleaved and rotated per rep, thread k pinned to the k-th CPU of --cpus,
//                               each thread walking its own experts out of --mb MB (more than L3: DRAM, as decode).
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kH = 2560, kFF = 640;
constexpr int kMaxNT = 64;

std::vector<std::string> split(const std::string& s, char c = ',') {
    std::vector<std::string> v;
    size_t from = 0;
    for (;;) {
        const size_t p = s.find(c, from);
        v.push_back(s.substr(from, p == std::string::npos ? std::string::npos : p - from));
        if (p == std::string::npos) break;
        from = p + 1;
    }
    return v;
}
std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    for (auto& x : split(s)) {
        const size_t dash = x.find('-');
        if (dash == std::string::npos) v.push_back(std::atoi(x.c_str()));
        else for (int i = std::atoi(x.substr(0, dash).c_str()); i <= std::atoi(x.substr(dash + 1).c_str()); ++i) v.push_back(i);
    }
    return v;
}
int type_of(const std::string& s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n != nullptr && s == n) return t;
    }
    return -1;
}

void pin(int c) {
#if defined(__linux__)
    if (c < 0) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
#else
    (void) c;
#endif
}

// the fp16 fields at the front of each block (d; and dmin / m for Q4_K / Q5_1): random values of every class but
// inf/nan, so the kernels' own fp16 conversion meets subnormals, zero and negatives
void scramble_scales(uint8_t* p, size_t bytes, int type, std::mt19937& rng) {
    const size_t bs = ggml_type_size((ggml_type) type);
    const int fields = type == 8 ? 1 : 2;
    for (size_t o = 0; o + bs <= bytes; o += bs)
        for (int k = 0; k < fields; ++k) {
            uint16_t h;
            const uint32_t r = rng();
            switch (r % 4) {
                case 0: h = (uint16_t) (r >> 16) & 0x83FFu; break;                       // subnormal or zero
                case 1: h = (uint16_t) (((r >> 16) % 0x7C00u) | (r & 0x8000u)); break;  // any finite
                default: { const ggml_fp16_t v = ggml_fp32_to_fp16(((r >> 8) % 1000) * 1e-5f * ((r & 4) ? -1.f : 1.f)); std::memcpy(&h, &v, 2); }
            }
            std::memcpy(p + o + 2 * k, &h, 2);
        }
}

void make_blob(const cpu::NativeFmt& f, uint8_t* blob, std::mt19937& rng, bool random_bytes) {
    if (random_bytes) {
        for (size_t i = 0; i < f.bytes; ++i) blob[i] = (uint8_t) rng();
        scramble_scales(blob, f.down_off, f.gu_type, rng);
        scramble_scales(blob + f.down_off, f.bytes - f.down_off, f.d_type, rng);
        return;
    }
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> row((size_t) std::max(kH, kFF));
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
    const auto* td = ggml_get_type_traits_cpu((ggml_type) f.d_type);
    for (int64_t r = 0; r < 2 * kFF; ++r) {
        for (int64_t i = 0; i < kH; ++i) row[(size_t) i] = nd(rng);
        tg->from_float(row.data(), blob + (size_t) r * f.gu_row, kH);
    }
    for (int64_t r = 0; r < kH; ++r) {
        for (int64_t i = 0; i < kFF; ++i) row[(size_t) i] = nd(rng);
        td->from_float(row.data(), blob + f.down_off + (size_t) r * f.d_row, kFF);
    }
}

// what native_gu_rows / native_down_rows run without the kernel: ggml's vec_dot per token
void old_gu(const cpu::NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff, int r0, int r1) {
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu((ggml_type) f.gu_type)->vec_dot;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot((int) f.n_embd, &g, 0, gr, 0, act[t], 0, 1);
            dot((int) f.n_embd, &u, 0, ur, 0, act[t], 0, 1);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}
void old_down(const cpu::NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out, int r0, int r1) {
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu((ggml_type) f.d_type)->vec_dot;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot((int) f.n_ff, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

struct Acts {
    std::vector<std::vector<uint8_t>> xa, ha;
    const void* act[kMaxNT];
    const void* hq[kMaxNT];
};
void make_acts(const cpu::NativeFmt& f, Acts& a, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    a.xa.assign(kMaxNT, std::vector<uint8_t>(cpu::kNativeActBytes));
    a.ha.assign(kMaxNT, std::vector<uint8_t>(cpu::kNativeHBytes));
    std::vector<float> x(kH), h(kFF);
    for (int t = 0; t < kMaxNT; ++t) {
        for (auto& v : x) v = nd(rng);
        for (auto& v : h) v = 0.1f * nd(rng);
        if (t % 7 == 3) for (auto& v : h) v *= 1e-4f;   // tiny activations: small fp16 activation scales too
        cpu::native_quant_act(f, x.data(), a.xa[t].data());
        cpu::native_quant_h(f, h.data(), a.ha[t].data());
        a.act[t] = a.xa[t].data();
        a.hq[t] = a.ha[t].data();
    }
}

struct Bufs {
    std::vector<float> ffb, outb;
    float* ff[kMaxNT];
    float* out[kMaxNT];
    Bufs() : ffb((size_t) kMaxNT * kFF, -7.f), outb((size_t) kMaxNT * kH, -7.f) {
        for (int t = 0; t < kMaxNT; ++t) { ff[t] = &ffb[(size_t) t * kFF]; out[t] = &outb[(size_t) t * kH]; }
    }
};

int parity(const std::vector<std::string>& pairs) {
    int failures = 0, checks = 0;
    double fast_rel = 0;   // the fast forms against ggml: the largest |diff| / max|row| seen
    const int nts[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64};
    for (const std::string& pair : pairs) {
        const size_t slash = pair.find('/');
        const int gt = type_of(pair.substr(0, slash)), dt = type_of(pair.substr(slash + 1));
        cpu::NativeFmt f;
        std::string err;
        if (gt < 0 || dt < 0 || !cpu::native_fmt(gt, dt, kH, kFF, f, err)) { std::printf("%s: %s\n", pair.c_str(), err.c_str()); return 1; }
        if (!cpu::kq128_supported(gt, kH) || !cpu::kq128_supported(dt, kFF)) { std::printf("%s: not a kq128 pair\n", pair.c_str()); return 1; }
        for (int variant = 0; variant < 2; ++variant) {
            std::mt19937 rng(99 + gt * 31 + dt + variant);
            std::vector<uint8_t> blob(f.bytes);
            make_blob(f, blob.data(), rng, variant == 1);
            Acts a;
            make_acts(f, a, rng);
            for (int nt : nts) {
                // whole matrices, then two partial row ranges (the pool's tasks), and the engine's dispatch
                const int ranges[3][4] = {{0, kFF, 0, kH}, {37, 101, 1000, 1003}, {kFF - 5, kFF, 0, 1}};
                for (int via = 0; via < (cpu::kq128_mode() == 2 ? 1 : 2); ++via)   // =fast: the dispatch is not ggml's bits
                    for (const auto& rg : ranges) {
                        Bufs o, n;
                        old_gu(f, blob.data(), a.act, nt, o.ff, rg[0], rg[1]);
                        old_down(f, blob.data(), a.hq, nt, o.out, rg[2], rg[3]);
                        if (via == 0) {
                            cpu::kq128_gu_rows(gt, blob.data(), f.gu_row, f.up_off, kH, a.act, nt, n.ff, rg[0], rg[1]);
                            cpu::kq128_rows(dt, blob.data() + f.down_off, f.d_row, kFF, a.hq, nt, n.out, rg[2], rg[3]);
                        } else {
                            cpu::native_gu_rows(f, blob.data(), a.act, nt, n.ff, rg[0], rg[1]);
                            cpu::native_down_rows(f, blob.data(), a.hq, nt, n.out, rg[2], rg[3]);
                        }
                        ++checks;
                        const bool gu_ok = std::memcmp(o.ffb.data(), n.ffb.data(), o.ffb.size() * 4) == 0;
                        const bool dn_ok = std::memcmp(o.outb.data(), n.outb.data(), o.outb.size() * 4) == 0;
                        if (!gu_ok || !dn_ok) {
                            ++failures;
                            size_t k = 0;
                            const bool g = !gu_ok;
                            const std::vector<float>& A = g ? o.ffb : o.outb;
                            const std::vector<float>& B = g ? n.ffb : n.outb;
                            while (k < A.size() && std::memcmp(&A[k], &B[k], 4) == 0) ++k;
                            std::printf("  MISMATCH %s %s nt %d rows %d-%d %s: [%zu] %.9g vs %.9g\n", pair.c_str(),
                                        variant ? "random-bytes" : "quantized", nt, g ? rg[0] : rg[2], g ? rg[1] : rg[3],
                                        via ? "native_*_rows" : "kq128", k, A[k], B[k]);
                        }
                    }
                // fast: a token's rows the same bits at every group size, and within float rounding of ggml's
                Bufs one, fa, o;
                old_gu(f, blob.data(), a.act, nt, o.ff, 0, kFF);
                old_down(f, blob.data(), a.hq, nt, o.out, 0, kH);
                cpu::kq128_gu_rows(gt, blob.data(), f.gu_row, f.up_off, kH, a.act, nt, fa.ff, 0, kFF, true);
                cpu::kq128_rows(dt, blob.data() + f.down_off, f.d_row, kFF, a.hq, nt, fa.out, 0, kH, true);
                for (int t = 0; t < nt; ++t) {
                    cpu::kq128_gu_rows(gt, blob.data(), f.gu_row, f.up_off, kH, a.act + t, 1, one.ff + t, 0, kFF, true);
                    cpu::kq128_rows(dt, blob.data() + f.down_off, f.d_row, kFF, a.hq + t, 1, one.out + t, 0, kH, true);
                }
                ++checks;
                if (std::memcmp(one.ffb.data(), fa.ffb.data(), one.ffb.size() * 4) != 0 ||
                    std::memcmp(one.outb.data(), fa.outb.data(), one.outb.size() * 4) != 0) {
                    ++failures;
                    std::printf("  FAST rows depend on the group size: %s nt %d\n", pair.c_str(), nt);
                }
                if (variant == 0) {   // ggml-quantized experts: compare per token, relative to the vector's largest value
                    for (int t = 0; t < nt; ++t) {
                        double mg = 0, dg = 0, md = 0, dd = 0;
                        for (int r = 0; r < kFF; ++r) { mg = std::max(mg, (double) std::fabs(o.ff[t][r])); dg = std::max(dg, (double) std::fabs(o.ff[t][r] - fa.ff[t][r])); }
                        for (int r = 0; r < kH; ++r) { md = std::max(md, (double) std::fabs(o.out[t][r])); dd = std::max(dd, (double) std::fabs(o.out[t][r] - fa.out[t][r])); }
                        fast_rel = std::max(fast_rel, std::max(dg / mg, dd / md));
                    }
                }
            }
        }
        std::printf("%s: checked (quantized + random-byte experts, nt 1..9/16/64, 3 row ranges, kernel + dispatch)\n", pair.c_str());
    }
    std::printf("kq_avx1_parity: fast forms vs ggml: max |diff| / max |value| = %.3g (limit 1e-5)\n", fast_rel);
    if (!(fast_rel <= 1e-5)) ++failures;
    std::printf(failures ? "kq_avx1_parity: %d of %d FAILED\n" : "kq_avx1_parity: all %d checks passed (exact rows bit-exact)\n",
                failures ? failures : checks, checks);
    return failures ? 1 : 0;
}

int bench(const std::vector<std::string>& pairs, const std::vector<int>& nts, const std::vector<int>& threads,
          const std::vector<int>& cpus, int mb, int reps, int part) {
    for (const std::string& pair : pairs) {
        const size_t slash = pair.find('/');
        const int gt = type_of(pair.substr(0, slash)), dt = type_of(pair.substr(slash + 1));
        cpu::NativeFmt f;
        std::string err;
        if (gt < 0 || dt < 0 || !cpu::native_fmt(gt, dt, kH, kFF, f, err)) { std::printf("%s: %s\n", pair.c_str(), err.c_str()); continue; }
        const size_t E = std::max<size_t>(64, ((size_t) mb << 20) / f.bytes);
        std::unique_ptr<uint8_t[]> mem(new uint8_t[E * f.bytes + 4096]);
        uint8_t* blobs = (uint8_t*) (((uintptr_t) mem.get() + 4095) & ~(uintptr_t) 4095);
        {
            std::mt19937 rng(1234 + gt * 64 + dt);
            const size_t U = std::min<size_t>(16, E);
            for (size_t e = 0; e < U; ++e) make_blob(f, blobs + e * f.bytes, rng, false);
            for (size_t e = U; e < E; ++e) std::memcpy(blobs + e * f.bytes, blobs + (e % U) * f.bytes, f.bytes);
        }
        std::mt19937 rng(77);
        Acts a;
        make_acts(f, a, rng);
        std::printf("\n%s: %zu experts of %.3f MB; us per expert = one thread's time for one whole expert (%s)\n",
                    pair.c_str(), E, f.bytes / 1e6, part == 1 ? "gate+up ONLY; GB/s counts the whole blob" : part == 2 ? "down ONLY; GB/s counts the whole blob" : "gate+up+down");
        for (int nt : nts) {
            for (int T : threads) {
                if (T < 1) continue;
                std::vector<double> gbs[3];
                for (int rep = 0; rep < reps; ++rep)
                    for (int k = 0; k < 3; ++k) {
                        const int arm = (rep + k) % 3;   // rotate which arm goes first
                        std::atomic<size_t> next{0};
                        auto work = [&](int idx) {
                            pin(cpus.empty() ? -1 : cpus[(size_t) idx % cpus.size()]);
                            Bufs b;
                            for (;;) {
                                const size_t e = next.fetch_add(1);
                                if (e >= E) break;
                                const uint8_t* blob = blobs + e * f.bytes;
                                if (arm == 0) {
                                    if (part != 2) old_gu(f, blob, a.act, nt, b.ff, 0, kFF);
                                    if (part != 1) old_down(f, blob, a.hq, nt, b.out, 0, kH);
                                } else {
                                    const bool fast = arm == 2;
                                    if (part != 2) cpu::kq128_gu_rows(gt, blob, f.gu_row, f.up_off, kH, a.act, nt, b.ff, 0, kFF, fast);
                                    if (part != 1) cpu::kq128_rows(dt, blob + f.down_off, f.d_row, kFF, a.hq, nt, b.out, 0, kH, fast);
                                }
                            }
                            volatile float sink = b.outb[0] + b.ffb[0];
                            (void) sink;
                        };
                        const auto t0 = Clock::now();
                        std::vector<std::thread> th;
                        for (int i = 0; i < T; ++i) th.emplace_back(work, i);
                        for (auto& x : th) x.join();
                        const double s = std::chrono::duration<double>(Clock::now() - t0).count();
                        gbs[arm].push_back(E * (double) f.bytes / s / 1e9);
                    }
                double med[3];
                for (int arm = 0; arm < 3; ++arm) {
                    std::sort(gbs[arm].begin(), gbs[arm].end());
                    med[arm] = gbs[arm][gbs[arm].size() / 2];
                }
                auto us = [&](double g) { return f.bytes / (g * 1e9 / T) * 1e6; };
                std::printf("  nt %2d threads %2d: old %6.2f GB/s %7.0f us | new %6.2f GB/s %7.0f us %.2fx | fast %6.2f GB/s %7.0f us %.2fx\n",
                            nt, T, med[0], us(med[0]), med[1], us(med[1]), med[1] / med[0], med[2], us(med[2]), med[2] / med[0]);
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool do_bench = false;
    std::vector<std::string> pairs = split("q4_K/q5_1,q8_0/q8_0,q4_K/q8_0");
    std::vector<int> nts = {1, 2, 4, 8, 64}, threads = {1, 8, 16}, cpus;
    int mb = 1024, reps = 5, part = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const bool more = i + 1 < argc;
        if (s == "--bench") do_bench = true;
        else if (s == "--pairs" && more) pairs = split(argv[++i]);
        else if (s == "--nt" && more) nts = ints(argv[++i]);
        else if (s == "--threads" && more) threads = ints(argv[++i]);
        else if (s == "--cpus" && more) cpus = ints(argv[++i]);
        else if (s == "--mb" && more) mb = std::atoi(argv[++i]);
        else if (s == "--reps" && more) reps = std::max(1, std::atoi(argv[++i]));
        else if (s == "--part" && more) { const std::string v = argv[++i]; part = v == "gu" ? 1 : v == "down" ? 2 : 0; }
        else {
            std::printf("usage: kq_avx1_parity [--bench [--pairs a/b,..] [--nt 1,2,4,8,64] [--threads 1,8,16] "
                        "[--cpus 10-19,30-39] [--mb 1024] [--reps 5]]\n");
            return 2;
        }
    }
    if (!cpu::cpu_avx1_ok()) { std::printf("kq_avx1_parity: no AVX here, the kernels would not run\n"); return 0; }
    ggml_cpu_init();
    std::printf("kq_avx1_parity: %s (avx2 %d)\n", cpu::cpu_name().c_str(), (int) cpu::cpu_avx2_ok());
    if (!do_bench && (ggml_cpu_has_avx2() || !ggml_cpu_has_avx())) {   // the reference is ggml's __AVX__ dot only
        std::printf("kq_avx1_parity: ggml-cpu is not the AVX-floor build here (STRATA_ISA_FLOOR=avx): nothing to compare\n");
        return 0;
    }
    for (auto& n : nts) n = std::min(std::max(n, 1), kMaxNT);
    return do_bench ? bench(pairs, nts, threads, cpus, mb, reps, part) : parity(pairs);
}
