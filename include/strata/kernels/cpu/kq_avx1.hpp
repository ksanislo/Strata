// include/strata/kernels/cpu/kq_avx1.hpp - the AVX1 router dot for older CPUs (from the Strata_Dirigo fork).
//
// bf16_rows_dot_multi (kq_avx2.cpp) is compiled with -mavx2 and uses instructions an AVX-only CPU does
// not have (`vpmovzxwd`, `vfmadd`), so an AVX-only host cannot call it at all.  This is the same
// arithmetic on AVX1 instructions: 8 float lanes per token, one weight stream per row shared by all
// `nt` tokens, one horizontal sum per (row, token).  AVX1 has no FMA, so the multiply and the add stay
// separate; it has no 256-bit integer ops, so the bf16 -> f32 widening runs in 128-bit halves.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Mirrors bf16_rows_dot_multi's arithmetic (same accumulation shape, mul+add instead of FMA).
/// Requires cols % 8 == 0 and nt <= 8, exactly like the AVX2 kernel it stands in for.
void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out);

/// Q4_K (12), Q5_1 (7) and Q8_0 (8) expert rows for any number of tokens, each token bit for bit ggml-cpu's
/// __AVX__ vec_dot (kq_avx1_parity checks): the weights are decoded once per block for all tokens.  `fast`: Q4_K and
/// Q5_1 with half the madds, within float rounding of ggml (Q8_0 unchanged); a token's row still does not depend
/// on how many tokens share the call.  kq128_supported (native_expert.cpp) says which type and row length fit.
bool kq128_supported(int ggml_type, int n) noexcept;
void kq128_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1, bool fast = false);
void kq128_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1, bool fast = false);

}  // namespace strata::kernels::cpu
