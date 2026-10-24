// include/strata/kernels/yarn_step.hpp - STRATA_YARN_ELASTIC: stored keys moved from one RoPE level to another.
//
// A key in the cache is post-RoPE: k' = R(theta_old(p)) k, its cos/sin carrying the old level's mscale.  Moving it
// to the new level is k'' = R(theta_new(p)) R(theta_old(p))^-1 k', per (position, pair), with no recompute.  `tab`
// holds, for positions [p0, p0 + n), 32 pairs x {c_old, s_old, c_new, s_new} - the two levels' table rows, so the
// keys end exactly as the tables the kernels read rotate new ones (device-readable: mapped pinned memory is fine).
#pragma once
#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// INT8 K (codes [page][kv_head][page_size][head_dim], one fp16 scale per 64 values): the cells of logical blocks
/// [b0, b1) inside [p0, p0 + n), the VRAM page of block b at page_table[b] (null: b itself; < 0: not resident, skipped).
/// Only the rotary group 0 (dims 0..63) changes; it is requantized as an append would.
void kv_q8_rerotate(int8_t* k_q, uint16_t* k_scale, const int32_t* page_table, int64_t b0, int64_t b1, int64_t p0,
                    int64_t n, const float* tab, const QsaShapes& s, void* stream);
/// The same for FP16 K pools ([page][kv_head][page_size][head_dim]).
void kv_f16_rerotate(uint16_t* k_pool, const int32_t* page_table, int64_t b0, int64_t b1, int64_t p0, int64_t n,
                     const float* tab, const QsaShapes& s, void* stream);
/// The plain rows of `tab` (c_old, s_old) for positions [p0, p0 + n), computed exactly as the RoPE kernels compute an
/// unscaled angle (an int position times powf(theta_scale, pair), rope_scaled_angle's identity constants).
void rope_plain_rows(float* tab, int64_t p0, int64_t n, float theta_scale, void* stream);
/// The indexer's pooled keys (fp32 [row][dim], rotary dims 0..63): rows [r0, r1), row r rotated at position 4r.
void idx_pooled_rerotate(float* pooled, int64_t r0, int64_t r1, int64_t p0, int64_t n, const float* tab, int dim,
                         void* stream);

}  // namespace strata::kernels
