// src/kernels/cpu/kq_avx1.cpp - the AVX1 router dot for older CPUs (see kq_avx1.hpp).  From rwkeyes' Strata_Dirigo
// fork (MIT), its AVX1 floor.
//
// Measured on the hardware this exists for (Xeon E5-2665, 2x8 cores, AVX only) for one layer's router
// [512 experts x 5120 embeddings]: the scalar std::fma fallback it replaces took 362 ms/layer, a scalar
// mul+add 4.9 ms, and this kernel 2.6 ms - 137x, with no extra memory (the weights stay bf16).
//
// Compiled for AVX (/arch:AVX, -mavx; CMakeLists.txt sets that per source); it is only ever called behind
// cpu_avx1_ok(), so no CPU that lacks AVX can reach it.
#include "strata/kernels/cpu/kq_avx1.hpp"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

#if !defined(__AVX__)
#error "kq_avx1.cpp must be compiled with AVX enabled (see the per-source flags in CMakeLists.txt)"
#endif

namespace strata::kernels::cpu {
namespace {

// 8 floats -> one float, the same reduction order the AVX2 kernel's hsum_float_8 uses.
inline float hsum_float_8_avx1(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

// 8 bf16 -> 8 f32: a bf16 is the top half of its f32, so widen in 128-bit halves (AVX1 has no
// 256-bit integer ops) and shift each 32-bit lane into the high half.
inline __m256 bf16x8_to_f32x8_avx1(const uint16_t* p) {
    const __m128i v = _mm_loadu_si128((const __m128i*) p);
    const __m128i z = _mm_setzero_si128();
    const __m128 lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_unpacklo_epi16(v, z), 16));
    const __m128 hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_unpackhi_epi16(v, z), 16));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}

}  // namespace

void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    // each row is read once for all `nt` (<= 8) tokens
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc[8];
        for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256 wf = bf16x8_to_f32x8_avx1(wr + c);
            for (int t = 0; t < nt; ++t) {
                const __m256 xv = _mm256_loadu_ps(x + (size_t) t * (size_t) cols + c);
                acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(wf, xv));   // no FMA on AVX1: mul, then add
            }
        }
        for (int t = 0; t < nt; ++t) {
            float s = hsum_float_8_avx1(acc[t]);
            for (; c < cols; ++c) {                                      // cols % 8 != 0 tail (cols % 8 == 0 today)
                const uint32_t bits = (uint32_t) wr[c] << 16;
                float wf; std::memcpy(&wf, &bits, sizeof wf);
                s += wf * x[(size_t) t * (size_t) cols + c];
            }
            out[(size_t) t * (size_t) rows + r] = s;
        }
    }
}

}  // namespace strata::kernels::cpu

// ---- The Unsloth / Q8_0 packs' expert rows (Q4_K, Q5_1, Q8_0) for several tokens at once, bit-exact against
// ggml-cpu's own __AVX__ dots (ggml-cpu/arch/x86/quants.c, the STRATA_ISA_FLOOR=avx build): kq_avx2.cpp's scheme in
// 128-bit integer lanes.  Everything that depends only on the weights (nibbles, fifth bits, 6-bit scales, |q8|, the
// fp16 scales) is done once per block; each token then runs ggml's integer chain and ggml's float operations in
// ggml's order, so every token's result is ggml's to the bit whatever the group size (the `Fast` forms excepted).
extern "C" float ggml_table_f32_f16[1 << 16];   // ggml-cpu.c

namespace strata::kernels::cpu {
namespace {

constexpr int kNT = 4;         // tokens per pass: accumulators stay in registers (16 xmm/ymm on x86-64)
constexpr int kMaxBlk = 256;   // 32-value blocks per row the per-call activation scale tables hold

// fp16 -> fp32 through ggml-cpu's own table (what GGML_CPU_FP16_TO_FP32 is without F16C; filled by ggml_cpu_init,
// which native_fmt runs before any expert reaches these rows): one load instead of a written-out conversion.
inline float h2f(uint16_t h) { return ggml_table_f32_f16[h]; }
inline float half_at(const void* block, int i) {
    uint16_t h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

// ggml's helpers (quants.c, __AVX__ forms)
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline __m256i set_m128i(__m128i hi, __m128i lo) { return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1); }

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX__).  Fast (STRATA_CPU_KERNELS=fast): the two 16-value halves of a
// scale's 32 values summed in int16 (|sum| <= 7680) before one madd, half the madds; the same integer total spread
// over the float lanes differently, so the result moves by float rounding only.
template <int NT, bool Fast>
void q4k_dot(const block_q4_K* x, int nb, const void* const* act, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);
    __m256 acc[NT];
    __m128 acc_m[NT];
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm256_setzero_ps();
        acc_m[t] = _mm_setzero_ps();
        y[t] = (const block_q8_K*) act[t];
    }
    for (int i = 0; i < nb; ++i) {
        const __m256 xd = _mm256_set1_ps(half_at(&x[i], 0));
        const __m128 nxdmin = _mm_set1_ps(-half_at(&x[i], 1));
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m128i utmps = _mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));
        // the block's weights, decoded once: per 64 values j, the low/high nibbles of two 16-byte halves
        __m128i sc_l[4], sc_h[4], q4l0[4], q4h0[4], q4l1[4], q4h1[4];
        __m128i shuffle = _mm_set1_epi16(0x0100);
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            sc_l[j] = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            sc_h[j] = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i b0 = _mm_loadu_si128((const __m128i*) q4);
            const __m128i b1 = _mm_loadu_si128((const __m128i*) (q4 + 16));
            q4 += 32;
            q4l0[j] = _mm_and_si128(b0, m4);
            q4h0[j] = _mm_and_si128(_mm_srli_epi16(b0, 4), m4);
            q4l1[j] = _mm_and_si128(b1, m4);
            q4h1[j] = _mm_and_si128(_mm_srli_epi16(b1, 4), m4);
        }
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* yb = y[t] + i;
            // ggml's d = y.d * x.d and dmin = -y.d * x.dmin, as lane products (the same values)
            const __m256 d = _mm256_mul_ps(_mm256_broadcast_ss(&yb->d), xd);
            const __m128 dmin = _mm_mul_ps(_mm_broadcast_ss(&yb->d), nxdmin);
            const __m128i q8s = _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) &yb->bsums[0]),
                                               _mm_loadu_si128((const __m128i*) &yb->bsums[8]));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_add_ps(_mm_mul_ps(dmin, _mm_cvtepi32_ps(prod)), acc_m[t]);
            const int8_t* q8 = yb->qs;
            __m128i s0 = _mm_setzero_si128(), s1 = _mm_setzero_si128();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m128i pl0 = _mm_maddubs_epi16(q4l0[j], _mm_loadu_si128((const __m128i*) q8));
                const __m128i pl1 = _mm_maddubs_epi16(q4l1[j], _mm_loadu_si128((const __m128i*) (q8 + 16)));
                const __m128i ph0 = _mm_maddubs_epi16(q4h0[j], _mm_loadu_si128((const __m128i*) (q8 + 32)));
                const __m128i ph1 = _mm_maddubs_epi16(q4h1[j], _mm_loadu_si128((const __m128i*) (q8 + 48)));
                q8 += 64;
                if (Fast) {
                    s0 = _mm_add_epi32(s0, _mm_madd_epi16(sc_l[j], _mm_add_epi16(pl0, pl1)));
                    s1 = _mm_add_epi32(s1, _mm_madd_epi16(sc_h[j], _mm_add_epi16(ph0, ph1)));
                } else {   // ggml's lanes: sumi_0 gets the first halves, sumi_1 the second (integer sums: any order)
                    s0 = _mm_add_epi32(s0, _mm_add_epi32(_mm_madd_epi16(sc_l[j], pl0), _mm_madd_epi16(sc_h[j], ph0)));
                    s1 = _mm_add_epi32(s1, _mm_add_epi32(_mm_madd_epi16(sc_l[j], pl1), _mm_madd_epi16(sc_h[j], ph1)));
                }
            }
            acc[t] = _mm256_add_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(set_m128i(s1, s0))), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// ---- Q5_1 x Q8_1 (ggml_vec_dot_q5_1_q8_1, __AVX__).  ys[t][ib] = {d, s} of the token's block, converted once.
// Fast: both halves' pair sums added in int16 (|sum| <= 15872), one madd and 4 float lanes instead of 8.
template <int NT, bool Fast>
void q5_1_dot(const block_q5_1* x, int nb, const float (*const* ys)[2], const void* const* act, float* res) {
    const __m128i mask = _mm_set1_epi8(0x10);
    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i shuf_l = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    const __m128i shuf_h = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
    const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
    const __m128i all1 = _mm_set1_epi64x(-1);
    __m256 acc[NT];
    float summs[NT];
    const block_q8_1* y[NT];
    __m128 acc4[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); acc4[t] = _mm_setzero_ps(); summs[t] = 0.0f; y[t] = (const block_q8_1*) act[t]; }
    for (int ib = 0; ib < nb; ++ib) {
        const float xm = half_at(&x[ib], 1);
        const __m256 dx = _mm256_set1_ps(half_at(&x[ib], 0));
        uint32_t x32;
        std::memcpy(&x32, x[ib].qh, 4);
        const __m128i bc = _mm_set1_epi32((int) x32);
        const __m128i hl = _mm_and_si128(_mm_cmpeq_epi8(_mm_or_si128(_mm_shuffle_epi8(bc, shuf_l), bit_mask), all1), mask);
        const __m128i hh = _mm_and_si128(_mm_cmpeq_epi8(_mm_or_si128(_mm_shuffle_epi8(bc, shuf_h), bit_mask), all1), mask);
        const __m128i raw = _mm_loadu_si128((const __m128i*) x[ib].qs);
        const __m128i bxl = _mm_or_si128(_mm_and_si128(raw, lowMask), hl);
        const __m128i bxh = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(raw, 4), lowMask), hh);
        for (int t = 0; t < NT; ++t) {
            summs[t] += xm * ys[t][ib][1];
            const __m128i* qy = (const __m128i*) y[t][ib].qs;
            const __m128i dotl = _mm_maddubs_epi16(bxl, _mm_loadu_si128(qy));
            const __m128i doth = _mm_maddubs_epi16(bxh, _mm_loadu_si128(qy + 1));
            if (Fast) {
                const __m128 q = _mm_cvtepi32_ps(_mm_madd_epi16(ones, _mm_add_epi16(dotl, doth)));
                acc4[t] = _mm_add_ps(_mm_mul_ps(q, _mm_mul_ps(_mm256_castps256_ps128(dx), _mm_broadcast_ss(&ys[t][ib][0]))), acc4[t]);
            } else {
                const __m256 q = _mm256_cvtepi32_ps(set_m128i(_mm_madd_epi16(ones, doth), _mm_madd_epi16(ones, dotl)));
                acc[t] = _mm256_add_ps(_mm256_mul_ps(q, _mm256_mul_ps(dx, _mm256_broadcast_ss(&ys[t][ib][0]))), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(Fast ? _mm256_insertf128_ps(acc[t], acc4[t], 0) : acc[t]) + summs[t];
}

// ---- Q8_0 x Q8_0 (ggml_vec_dot_q8_0_q8_0, __AVX__: blocks in pairs, mul_sum_i8_quad_float, quad_fp16_delta_float)
// yp[t][ib / 2] = the token's scales of blocks ib, ib + 1 as {d0 x4, d1 x4}; yd[t][ib] = one block's (the odd tail).
template <int NT>
void q8_0_dot(const block_q8_0* x, int nb, const __m256* const* yp, const float* const* yd, const void* const* act,
              float* res) {
    const __m128i mone = _mm_set1_epi16(1);
    __m256 acc[NT];
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); y[t] = (const block_q8_0*) act[t]; }
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m128i x10 = _mm_loadu_si128((const __m128i*) x[ib].qs);
        const __m128i x11 = _mm_loadu_si128((const __m128i*) x[ib].qs + 1);
        const __m128i x20 = _mm_loadu_si128((const __m128i*) x[ib + 1].qs);
        const __m128i x21 = _mm_loadu_si128((const __m128i*) x[ib + 1].qs + 1);
        const __m128i a10 = _mm_sign_epi8(x10, x10), a11 = _mm_sign_epi8(x11, x11);
        const __m128i a20 = _mm_sign_epi8(x20, x20), a21 = _mm_sign_epi8(x21, x21);
        const __m256 xdp = _mm256_set_m128(_mm_set1_ps(half_at(&x[ib + 1], 0)), _mm_set1_ps(half_at(&x[ib], 0)));
        for (int t = 0; t < NT; ++t) {
            const __m128i* q1 = (const __m128i*) y[t][ib].qs;
            const __m128i* q2 = (const __m128i*) y[t][ib + 1].qs;
            const __m128i p10 = _mm_madd_epi16(_mm_maddubs_epi16(a10, _mm_sign_epi8(_mm_loadu_si128(q1), x10)), mone);
            const __m128i p11 = _mm_madd_epi16(_mm_maddubs_epi16(a11, _mm_sign_epi8(_mm_loadu_si128(q1 + 1), x11)), mone);
            const __m128i p20 = _mm_madd_epi16(_mm_maddubs_epi16(a20, _mm_sign_epi8(_mm_loadu_si128(q2), x20)), mone);
            const __m128i p21 = _mm_madd_epi16(_mm_maddubs_epi16(a21, _mm_sign_epi8(_mm_loadu_si128(q2 + 1), x21)), mone);
            const __m256 p = _mm256_cvtepi32_ps(set_m128i(_mm_add_epi32(p20, p21), _mm_add_epi32(p10, p11)));
            const __m256 deltas = _mm256_mul_ps(xdp, yp[t][ib / 2]);   // quad_fp16_delta_float's lane products
            acc[t] = _mm256_add_ps(_mm256_mul_ps(deltas, p), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        float sumf = hsum_float_8(acc[t]);
        for (int b = ib; b < nb; ++b) {   // ggml's scalar tail (an odd block count)
            int sumi = 0;
            for (int j = 0; j < QK8_0; ++j) sumi += x[b].qs[j] * y[t][b].qs[j];
            sumf += sumi * (half_at(&x[b], 0) * yd[t][b]);
        }
        res[t] = sumf;
    }
}

// One pass of up to kNT tokens over rows [r0, r1) of one matrix (two for gate/up: then SwiGLU into ff).
template <int NT, bool Fast>
void rows_pass(int type, const uint8_t* w, size_t row_bytes, size_t up_off, bool gu, int n, const void* const* act,
               float* const* out, int r0, int r1) {
    // the activations' fp16 scales, converted once per call instead of once per row
    float ysc[NT][kMaxBlk][2];
    float yd8[NT][kMaxBlk];
    __m256 ypr[NT][kMaxBlk / 2];
    const float (*ys[NT])[2];
    const float* yd[NT];
    const __m256* yp[NT];
    const int nb32 = n / 32;
    for (int t = 0; t < NT; ++t) {
        ys[t] = ysc[t];
        yd[t] = yd8[t];
        yp[t] = ypr[t];
        if (type == 7)
            for (int b = 0; b < nb32; ++b) {
                ysc[t][b][0] = half_at((const block_q8_1*) act[t] + b, 0);
                ysc[t][b][1] = half_at((const block_q8_1*) act[t] + b, 1);
            }
        else if (type == 8) {
            for (int b = 0; b < nb32; ++b) yd8[t][b] = half_at((const block_q8_0*) act[t] + b, 0);
            for (int b = 0; b + 1 < nb32; b += 2) ypr[t][b / 2] = _mm256_set_m128(_mm_set1_ps(yd8[t][b + 1]), _mm_set1_ps(yd8[t][b]));
        }
    }
    auto dot = [&](const uint8_t* row, float* res) {
        switch (type) {
            case 12: q4k_dot<NT, Fast>((const block_q4_K*) row, n / QK_K, act, res); break;
            case 7: q5_1_dot<NT, Fast>((const block_q5_1*) row, nb32, ys, act, res); break;
            default: q8_0_dot<NT>((const block_q8_0*) row, nb32, yp, yd, act, res); break;
        }
    };
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        dot(w + (size_t) r * row_bytes, g);
        if (gu) {
            dot(w + up_off + (size_t) r * row_bytes, u);
            // native_gu_rows' own SwiGLU expression: the same bits as the per-token ggml path
            for (int t = 0; t < NT; ++t) out[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        } else {
            for (int t = 0; t < NT; ++t) out[t][r] = g[t];
        }
    }
}

template <bool Fast>
void rows_any(int type, const uint8_t* w, size_t row_bytes, size_t up_off, bool gu, int n, const void* const* act,
              int nt, float* const* out, int r0, int r1) {
    // more than kNT tokens: every pass over a block of rows small enough for L1/L2, so the weights come from DRAM once
    const int rb = nt <= kNT ? r1 - r0 : (int) ((16384 + row_bytes - 1) / row_bytes);
    for (int a = r0; a < r1; a += rb) {
        const int b = r1 - a < rb ? r1 : a + rb;
        for (int t0 = 0; t0 < nt; t0 += kNT) {
            const int k = nt - t0 < kNT ? nt - t0 : kNT;
            switch (k) {
                case 1: rows_pass<1, Fast>(type, w, row_bytes, up_off, gu, n, act + t0, out + t0, a, b); break;
                case 2: rows_pass<2, Fast>(type, w, row_bytes, up_off, gu, n, act + t0, out + t0, a, b); break;
                case 3: rows_pass<3, Fast>(type, w, row_bytes, up_off, gu, n, act + t0, out + t0, a, b); break;
                default: rows_pass<4, Fast>(type, w, row_bytes, up_off, gu, n, act + t0, out + t0, a, b); break;
            }
        }
    }
    _mm256_zeroupper();
}

}  // namespace

void kq128_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1, bool fast) {
    if (fast) rows_any<true>(type, blob, gu_row, up_off, true, n, act, nt, ff, r0, r1);
    else rows_any<false>(type, blob, gu_row, up_off, true, n, act, nt, ff, r0, r1);
}

void kq128_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1, bool fast) {
    if (fast) rows_any<true>(type, w, row_bytes, 0, false, n, act, nt, out, r0, r1);
    else rows_any<false>(type, w, row_bytes, 0, false, n, act, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu
