// src/kernels/cuda/yarn_step.cu - see include/strata/kernels/yarn_step.hpp.
#include "strata/kernels/yarn_step.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int ROT = 64, HALF = 32;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "yarn_step: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// the rotary position of `cell`'s pair i: its index, or a picture's (t, h, w) as the M-RoPE kernels read them
__device__ __forceinline__ long long pair_pos(const int32_t* thw, long long p0, long long cell, int i) {
    return thw ? (long long) thw[(cell - p0) * 3 + i % 3] : cell;
}

// x[0..63] in shared memory, thread t: the pair (t % 32, t % 32 + 32) of `cell` moved to the new level (tab's rows
// from position tb on); returns its new value
__device__ __forceinline__ float rerotate_one(const float* x, const float* tab, long long tb, const int32_t* thw,
                                              long long p0, long long cell, int t) {
    const int i = t % HALF;
    const float* row = tab + (pair_pos(thw, p0, cell, i) - tb) * HALF * 4;
    const float c0 = row[i * 4], s0 = row[i * 4 + 1], c1 = row[i * 4 + 2], s1 = row[i * 4 + 3];
    const float a = x[i], b = x[i + HALF];
    const float r = c0 * c0 + s0 * s0;   // the old mscale squared
    const float a0 = (c0 * a + s0 * b) / r, b0 = (c0 * b - s0 * a) / r;
    return t < HALF ? c1 * a0 - s1 * b0 : s1 * a0 + c1 * b0;
}

// one block per (cell, kv head), 64 threads: group 0 dequantized, moved, requantized against its new stored scale
__global__ void q8_kernel(int8_t* __restrict__ k_q, uint16_t* __restrict__ k_scale, const int32_t* __restrict__ table,
                          long long c0, long long p0, const float* __restrict__ tab, long long tb,
                          const int32_t* __restrict__ thw, int kv_heads, int head_dim, int page_size) {
    const long long cell = c0 + blockIdx.x;
    const int h = blockIdx.y, t = threadIdx.x;
    const long long page = table ? (long long) table[cell / page_size] : cell / page_size;
    if (page < 0) return;
    const int groups = head_dim / KV_Q8_GROUP;
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    __shared__ float x[ROT];
    __shared__ float warp_max[2];
    x[t] = (float) k_q[row * head_dim + t] * f32_from_f16(k_scale[row * groups]);
    __syncthreads();
    const float y = rerotate_one(x, tab, tb, thw, p0, cell, t);
    float a = fabsf(y);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((t & 31) == 0) warp_max[t >> 5] = a;
    __syncthreads();
    const uint16_t sbits = f16_from_f32(fmaxf(warp_max[0], warp_max[1]) / 127.0f);
    const float sf = f32_from_f16(sbits);
    int q = 0;
    if (sf > 0.0f) {
        q = __float2int_rn(y / sf);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    k_q[row * head_dim + t] = (int8_t) q;
    if (t == 0) k_scale[row * groups] = sbits;
}

__global__ void f16_kernel(uint16_t* __restrict__ k_pool, const int32_t* __restrict__ table, long long c0, long long p0,
                           const float* __restrict__ tab, long long tb, const int32_t* __restrict__ thw, int kv_heads,
                           int head_dim, int page_size) {
    const long long cell = c0 + blockIdx.x;
    const int h = blockIdx.y, t = threadIdx.x;
    const long long page = table ? (long long) table[cell / page_size] : cell / page_size;
    if (page < 0) return;
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    __shared__ float x[ROT];
    x[t] = f32_from_f16(k_pool[row * head_dim + t]);
    __syncthreads();
    const float y = rerotate_one(x, tab, tb, thw, p0, cell, t);
    k_pool[row * head_dim + t] = f16_from_f32(y);
}

__global__ void pooled_kernel(float* __restrict__ pooled, long long r0, long long p0, const float* __restrict__ tab,
                              long long tb, const int32_t* __restrict__ thw, int dim) {
    const long long r = r0 + blockIdx.x;
    const int t = threadIdx.x;
    __shared__ float x[ROT];
    x[t] = pooled[r * dim + t];
    __syncthreads();
    const float y = rerotate_one(x, tab, tb, thw, p0, 4 * r, t);
    pooled[r * dim + t] = y;
}

__global__ void plain_rows_kernel(float* __restrict__ tab, long long p0, float theta_scale) {
    const int pos = (int) (p0 + blockIdx.x), pair = threadIdx.x;
    const float theta_extrap = pos * powf(theta_scale, float(pair));
    float c, s;
    rope_scaled_angle(theta_extrap, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, pair, c, s);
    tab[(size_t) blockIdx.x * HALF * 4 + pair * 4] = c;
    tab[(size_t) blockIdx.x * HALF * 4 + pair * 4 + 1] = s;
}

// the cells of blocks [b0, b1) that [p0, p0 + n) holds
bool cell_range(int64_t b0, int64_t b1, int64_t p0, int64_t n, int64_t page_size, int64_t& c0, int64_t& c1) {
    c0 = std::max<int64_t>(b0 * page_size, p0);
    c1 = std::min<int64_t>(b1 * page_size, p0 + n);
    return c1 > c0;
}

}  // namespace

void kv_q8_rerotate(int8_t* k_q, uint16_t* k_scale, const int32_t* page_table, int64_t b0, int64_t b1, int64_t p0,
                    int64_t n, const float* tab, const QsaShapes& s, void* stream, const int32_t* thw, int64_t tb) {
    int64_t c0 = 0, c1 = 0;
    if (s.n_rot != ROT || s.head_dim % KV_Q8_GROUP != 0 || !cell_range(b0, b1, p0, n, s.page_size, c0, c1)) return;
    q8_kernel<<<dim3((unsigned) (c1 - c0), (unsigned) s.n_head_kv), ROT, 0, (cudaStream_t) stream>>>(
        k_q, k_scale, page_table, c0, p0, tab, tb < 0 ? p0 : tb, thw, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size);
    check("q8");
}

void kv_f16_rerotate(uint16_t* k_pool, const int32_t* page_table, int64_t b0, int64_t b1, int64_t p0, int64_t n,
                     const float* tab, const QsaShapes& s, void* stream, const int32_t* thw, int64_t tb) {
    int64_t c0 = 0, c1 = 0;
    if (s.n_rot != ROT || !cell_range(b0, b1, p0, n, s.page_size, c0, c1)) return;
    f16_kernel<<<dim3((unsigned) (c1 - c0), (unsigned) s.n_head_kv), ROT, 0, (cudaStream_t) stream>>>(
        k_pool, page_table, c0, p0, tab, tb < 0 ? p0 : tb, thw, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size);
    check("f16");
}

void rope_plain_rows(float* tab, int64_t p0, int64_t n, float theta_scale, void* stream) {
    if (n <= 0) return;
    plain_rows_kernel<<<(unsigned) n, HALF, 0, (cudaStream_t) stream>>>(tab, p0, theta_scale);
    check("plain rows");
}

void idx_pooled_rerotate(float* pooled, int64_t r0, int64_t r1, int64_t p0, int64_t n, const float* tab, int dim,
                         void* stream, const int32_t* thw, int64_t tb) {
    // the rows whose cell 4r falls in [p0, p0 + n)
    const int64_t a = std::max<int64_t>(r0, (p0 + 3) / 4), b = std::min<int64_t>(r1, (p0 + n + 3) / 4);
    if (b <= a) return;
    pooled_kernel<<<(unsigned) (b - a), ROT, 0, (cudaStream_t) stream>>>(pooled, a, p0, tab, tb < 0 ? p0 : tb, thw, dim);
    check("pooled");
}

}  // namespace strata::kernels
