// Q8_0 integer quantisation: blocks of 32 values along the input dimension share one float32 scale.
//
// Quantising a block: amax = max |x_i| (float), d = amax / 127 (float division), q_i = round(x_i * (1 / d)) with
// round-half-to-even implemented explicitly (independent of the floating point environment), clamped to +-127.
// A block of zeros (or with non-finite values) gets d = 0 and q = 0 wherever x is not finite.
//
// linear_q8: activations are quantised per row with the same rule, then for every output
//   acc = sum over blocks b ascending of (double(dx_b) * double(dw_b)) * double(isum_b),   isum_b = sum_i xq_i wq_i
// where isum_b is an exact int32 (32 * 127 * 127 < 2^31), so its evaluation order is irrelevant and the compiler may
// vectorise it freely; the double combine runs over blocks ascending, then the bias is added and the result is
// rounded to float once. Outputs are grouped in panels of 16 that are the thread pool's tasks.
#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "dllm/parallel.hpp"
#include "dllm/simd.hpp"

namespace dllm {

constexpr std::size_t kQ8Block = 32;

// Round half to even for |v| <= 2^22, computed with exact operations only.
inline float round_half_even(float v) {
    const float f = std::floor(v);
    const float diff = v - f;  // exact: v and f are within one unit of each other
    if (diff > 0.5f) {
        return f + 1.0f;
    }
    if (diff < 0.5f) {
        return f;
    }
    return std::fmod(f, 2.0f) == 0.0f ? f : f + 1.0f;
}

// Quantises n values (n a multiple of kQ8Block) into q[n] and scales[n / kQ8Block].
inline void quantize_q8_0(const float* x, std::int8_t* q, float* scales, std::size_t n) {
    for (std::size_t b = 0; b * kQ8Block < n; ++b) {
        const float* xb = x + b * kQ8Block;
        float amax = 0.0f;
        for (std::size_t i = 0; i < kQ8Block; ++i) {
            const float a = std::fabs(xb[i]);
            if (a > amax && std::isfinite(a)) {
                amax = a;
            }
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scales[b] = d;
        for (std::size_t i = 0; i < kQ8Block; ++i) {
            float v = std::isfinite(xb[i]) ? round_half_even(xb[i] * id) : 0.0f;
            v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
            q[b * kQ8Block + i] = static_cast<std::int8_t>(v);
        }
    }
}

// Q4_0: blocks of 32 values share a float32 scale, like Q8_0, with 4-bit values. Quantising a block:
// amax = max |x_i| (float), d = amax / 7 (float division), q_i = round(x_i * (1 / d)) with the round-half-to-even of
// Q8_0, clamped to +-7. A block is stored in 16 bytes: byte j holds q_j + 8 in its low and q_(j+16) + 8 in its high
// four bits. linear_q4 unpacks the values to int8 and then runs exactly the Q8_0 computation (activations quantised
// to Q8_0, exact int32 block sums, combined in double over blocks ascending), so its order is that of linear_q8.
constexpr std::size_t kQ4BlockBytes = kQ8Block / 2;

// Quantises n values (n a multiple of kQ8Block) into packed[n / 2] and scales[n / kQ8Block].
inline void quantize_q4_0(const float* x, std::uint8_t* packed, float* scales, std::size_t n) {
    for (std::size_t b = 0; b * kQ8Block < n; ++b) {
        const float* xb = x + b * kQ8Block;
        float amax = 0.0f;
        for (std::size_t i = 0; i < kQ8Block; ++i) {
            const float a = std::fabs(xb[i]);
            if (a > amax && std::isfinite(a)) {
                amax = a;
            }
        }
        const float d = amax / 7.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scales[b] = d;
        int q[kQ8Block];
        for (std::size_t i = 0; i < kQ8Block; ++i) {
            float v = std::isfinite(xb[i]) ? round_half_even(xb[i] * id) : 0.0f;
            v = v > 7.0f ? 7.0f : (v < -7.0f ? -7.0f : v);
            q[i] = static_cast<int>(v);
        }
        for (std::size_t j = 0; j < kQ4BlockBytes; ++j) {
            packed[b * kQ4BlockBytes + j] = static_cast<std::uint8_t>((q[j] + 8) | ((q[j + kQ4BlockBytes] + 8) << 4));
        }
    }
}

// The int8 values of n packed Q4_0 values (n a multiple of kQ8Block).
inline void unpack_q4_0(const std::uint8_t* packed, std::int8_t* values, std::size_t n) {
    for (std::size_t b = 0; b * kQ8Block < n; ++b) {
        const std::uint8_t* src = packed + b * kQ4BlockBytes;
        std::int8_t* dst = values + b * kQ8Block;
        for (std::size_t j = 0; j < kQ4BlockBytes; ++j) {
            dst[j] = static_cast<std::int8_t>((src[j] & 0x0F) - 8);
            dst[j + kQ4BlockBytes] = static_cast<std::int8_t>((src[j] >> 4) - 8);
        }
    }
}

// Exact int32 sum of the 32 products of two int8 blocks (values in [-128, 127]): SSE2 on x86-64, NEON on arm64.
inline std::int32_t q8_block_dot(const std::int8_t* xb, const std::int8_t* wb) {
#if defined(DLLM_SSE2)
    __m128i sum = _mm_setzero_si128();
    for (std::size_t h = 0; h < kQ8Block; h += 16) {
        const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xb + h));
        const __m128i w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wb + h));
        // Sign-extend bytes to 16 bits (unpack into the high byte, then shift right arithmetically).
        const __m128i xl = _mm_srai_epi16(_mm_unpacklo_epi8(x, x), 8);
        const __m128i xh = _mm_srai_epi16(_mm_unpackhi_epi8(x, x), 8);
        const __m128i wl = _mm_srai_epi16(_mm_unpacklo_epi8(w, w), 8);
        const __m128i wh = _mm_srai_epi16(_mm_unpackhi_epi8(w, w), 8);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(xl, wl));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(xh, wh));
    }
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4e));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb1));
    return _mm_cvtsi128_si32(sum);
#elif defined(DLLM_NEON)
    int32x4_t sum = vdupq_n_s32(0);
    for (std::size_t h = 0; h < kQ8Block; h += 16) {
        const int8x16_t x = vld1q_s8(xb + h);
        const int8x16_t w = vld1q_s8(wb + h);
        sum = vpadalq_s16(sum, vmull_s8(vget_low_s8(x), vget_low_s8(w)));
        sum = vpadalq_s16(sum, vmull_s8(vget_high_s8(x), vget_high_s8(w)));
    }
    return vaddvq_s32(sum);
#else
    std::int32_t sum = 0;
    for (std::size_t i = 0; i < kQ8Block; ++i) {
        sum += static_cast<std::int32_t>(xb[i]) * static_cast<std::int32_t>(wb[i]);
    }
    return sum;
#endif
}

// Rows [0, rows) of the outputs [n0, n0 + 16) of a Q8_0 linear layer, with the given exact block dot product. wq and
// ws point at the panel's first output (row n0 of the weights and scales).
template <std::int32_t (*Dot)(const std::int8_t*, const std::int8_t*)>
DLLM_ALWAYS_INLINE void linear_q8_panel(const std::int8_t* xq, const float* xs, const std::int8_t* wq,
                                        const float* ws, const float* bias, float* out, std::size_t rows,
                                        std::size_t in_features, std::size_t out_features, std::size_t n0) {
    const std::size_t width = n0 + 16 <= out_features ? 16 : out_features - n0;
    const std::size_t blocks = in_features / kQ8Block;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::int8_t* xr = xq + r * in_features;
        const float* xsr = xs + r * blocks;
        for (std::size_t j = 0; j < width; ++j) {
            const std::size_t n = n0 + j;
            const std::int8_t* wn = wq + j * in_features;
            const float* wsn = ws + j * blocks;
            double acc = 0.0;
            for (std::size_t b = 0; b < blocks; ++b) {
                const std::int32_t isum = Dot(xr + b * kQ8Block, wn + b * kQ8Block);
                acc += (static_cast<double>(xsr[b]) * static_cast<double>(wsn[b])) * static_cast<double>(isum);
            }
            if (bias != nullptr) {
                acc += bias[n];
            }
            out[r * out_features + n] = static_cast<float>(acc);
        }
    }
}

using LinearQ8PanelFn = void (*)(const std::int8_t*, const float*, const std::int8_t*, const float*, const float*,
                                 float*, std::size_t, std::size_t, std::size_t, std::size_t);

inline void linear_q8_panel_portable(const std::int8_t* xq, const float* xs, const std::int8_t* wq, const float* ws,
                                     const float* bias, float* out, std::size_t rows, std::size_t in_features,
                                     std::size_t out_features, std::size_t n0) {
    linear_q8_panel<q8_block_dot>(xq, xs, wq, ws, bias, out, rows, in_features, out_features, n0);
}

#ifdef DLLM_X86_DISPATCH
// |w| times sign-adjusted x as u8 x s8 pairs (a pair sum is at most 2 * 128 * 127 < 2^15, so the 16-bit step
// cannot saturate; x never holds -128 because quantize_q8_0 clamps to +-127), widened to eight int32 partial sums.
DLLM_TARGET_AVX2 inline __m256i q8_partial_sums_avx2(__m256i x, __m256i w) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)),
                             _mm256_set1_epi16(1));
}

// The exact int32 totals of four vectors of partial sums, as one vector [sum q0, sum q1, sum q2, sum q3].
DLLM_TARGET_AVX2 inline __m128i q8_totals_avx2(__m256i q0, __m256i q1, __m256i q2, __m256i q3) {
    const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(q0, q1), _mm256_hadd_epi32(q2, q3));
    return _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
}

// unpack_q4_0 with AVX2: 16 bytes become the 32 int8 values of a block.
DLLM_TARGET_AVX2 inline void unpack_q4_0_avx2(const std::uint8_t* packed, std::int8_t* values, std::size_t n) {
    for (std::size_t b = 0; b * kQ8Block < n; ++b) {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + b * kQ4BlockBytes));
        const __m128i low = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
        const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0F));
        const __m256i w = _mm256_sub_epi8(_mm256_set_m128i(high, low), _mm256_set1_epi8(8));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(values + b * kQ8Block), w);
    }
}

struct Q8TileArgs {
    const std::int8_t* xq;     // the tile's first row
    const float* xs;           // its scales
    const std::int8_t* w[4];   // the four outputs' weight rows
    const double* scales;      // their scales, [blocks][4]
    std::size_t in_features;
    std::size_t blocks;
};

// R rows by four outputs (R = 1 or 2): each weight block loaded serves R rows, and the four block sums of a row
// are reduced together. Per output the double combine is that of linear_q8_panel(): the product of the two scales,
// times the block sum, added over blocks ascending, as separate multiplies and adds (no fused multiply-add).
template <std::size_t R>
DLLM_TARGET_AVX2 inline void q8_tile_avx2(const Q8TileArgs& a, double (&result)[R][4]) {
    __m256d acc[R];
    for (std::size_t i = 0; i < R; ++i) {
        acc[i] = _mm256_setzero_pd();
    }
    for (std::size_t b = 0; b < a.blocks; ++b) {
        __m256i x[R];
        for (std::size_t i = 0; i < R; ++i) {
            x[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.xq + i * a.in_features + b * kQ8Block));
        }
        __m256i q[R][4];
        for (std::size_t j = 0; j < 4; ++j) {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.w[j] + b * kQ8Block));
            for (std::size_t i = 0; i < R; ++i) {
                q[i][j] = q8_partial_sums_avx2(x[i], w);
            }
        }
        const __m256d dw = _mm256_loadu_pd(a.scales + b * 4);
        for (std::size_t i = 0; i < R; ++i) {
            const __m256d sums = _mm256_cvtepi32_pd(q8_totals_avx2(q[i][0], q[i][1], q[i][2], q[i][3]));
            const __m256d scale = _mm256_mul_pd(_mm256_set1_pd(static_cast<double>(a.xs[i * a.blocks + b])), dw);
            acc[i] = _mm256_add_pd(acc[i], _mm256_mul_pd(scale, sums));
        }
    }
    for (std::size_t i = 0; i < R; ++i) {
        _mm256_storeu_pd(result[i], acc[i]);
    }
}

template <bool Q4>
DLLM_TARGET_AVX2 inline void linear_quant_panel_avx2(const std::int8_t* xq, const float* xs, const std::uint8_t* wq,
                                                     const float* ws, const float* bias, float* out, std::size_t rows,
                                                     std::size_t in_features, std::size_t out_features,
                                                     std::size_t n0) {
    const std::size_t row_bytes = Q4 ? in_features / 2 : in_features;
    const std::size_t width = n0 + 16 <= out_features ? 16 : out_features - n0;
    const std::size_t blocks = in_features / kQ8Block;
    thread_local std::vector<double> scales;
    thread_local std::vector<std::int8_t> unpacked;  // Q4: the four outputs' values as int8
    scales.resize(blocks * 4);
    unpacked.resize(Q4 ? 4 * in_features : 0);
    for (std::size_t g = 0; g < width; g += 4) {
        Q8TileArgs args{nullptr, nullptr, {}, scales.data(), in_features, blocks};
        for (std::size_t j = 0; j < 4; ++j) {
            const std::size_t k = g + j < width ? g + j : g;  // past the last output: repeat one, unused
            if constexpr (Q4) {
                unpack_q4_0_avx2(wq + k * row_bytes, unpacked.data() + j * in_features, in_features);
                args.w[j] = unpacked.data() + j * in_features;
            } else {
                args.w[j] = reinterpret_cast<const std::int8_t*>(wq + k * row_bytes);
            }
            for (std::size_t b = 0; b < blocks; ++b) {
                scales[b * 4 + j] = ws[k * blocks + b];
            }
        }
        const std::size_t end = g + 4 < width ? g + 4 : width;
        for (std::size_t r = 0; r < rows; r += 2) {
            args.xq = xq + r * in_features;
            args.xs = xs + r * blocks;
            double result[2][4];
            const std::size_t count = r + 2 <= rows ? 2 : 1;
            if (count == 2) {
                q8_tile_avx2<2>(args, result);
            } else {
                double single[1][4];
                q8_tile_avx2<1>(args, single);
                std::copy(single[0], single[0] + 4, result[0]);
            }
            for (std::size_t i = 0; i < count; ++i) {
                for (std::size_t j = g; j < end; ++j) {
                    double acc = result[i][j - g];
                    if (bias != nullptr) {
                        acc += bias[n0 + j];
                    }
                    out[(r + i) * out_features + n0 + j] = static_cast<float>(acc);
                }
            }
        }
    }
}

DLLM_TARGET_AVX2 inline void linear_q8_panel_avx2(const std::int8_t* xq, const float* xs, const std::int8_t* wq,
                                                  const float* ws, const float* bias, float* out, std::size_t rows,
                                                  std::size_t in_features, std::size_t out_features,
                                                  std::size_t n0) {
    linear_quant_panel_avx2<false>(xq, xs, reinterpret_cast<const std::uint8_t*>(wq), ws, bias, out, rows,
                                   in_features, out_features, n0);
}

DLLM_TARGET_AVX2 inline void linear_q4_panel_avx2(const std::int8_t* xq, const float* xs, const std::uint8_t* wq,
                                                  const float* ws, const float* bias, float* out, std::size_t rows,
                                                  std::size_t in_features, std::size_t out_features,
                                                  std::size_t n0) {
    linear_quant_panel_avx2<true>(xq, xs, wq, ws, bias, out, rows, in_features, out_features, n0);
}
#endif

inline LinearQ8PanelFn linear_q8_panel_kernel() {
#ifdef DLLM_X86_DISPATCH
    switch (active_isa()) {
        case Isa::avx2:
            return linear_q8_panel_avx2;
        default:
            break;
    }
#endif
    return linear_q8_panel_portable;
}

using LinearQ4PanelFn = void (*)(const std::int8_t*, const float*, const std::uint8_t*, const float*, const float*,
                                 float*, std::size_t, std::size_t, std::size_t, std::size_t);

// The baseline Q4_0 panel: the panel's weights unpacked to int8, then the baseline Q8_0 panel.
inline void linear_q4_panel_portable(const std::int8_t* xq, const float* xs, const std::uint8_t* wq, const float* ws,
                                     const float* bias, float* out, std::size_t rows, std::size_t in_features,
                                     std::size_t out_features, std::size_t n0) {
    const std::size_t width = n0 + 16 <= out_features ? 16 : out_features - n0;
    thread_local std::vector<std::int8_t> panel;
    panel.resize(16 * in_features);
    unpack_q4_0(wq, panel.data(), width * in_features);
    linear_q8_panel_portable(xq, xs, panel.data(), ws, bias, out, rows, in_features, out_features, n0);
}

inline LinearQ4PanelFn linear_q4_panel_kernel() {
#ifdef DLLM_X86_DISPATCH
    switch (active_isa()) {
        case Isa::avx2:
            return linear_q4_panel_avx2;
        default:
            break;
    }
#endif
    return linear_q4_panel_portable;
}

// Quantises the activations x[rows][in_features] for the Q8_0 and Q4_0 kernels, rows in blocks on the pool.
inline void quantize_rows_q8_0(const float* x, std::int8_t* xq, float* xs, std::size_t rows, std::size_t in_features) {
    constexpr std::size_t kRows = 64;
    const std::size_t blocks = in_features / kQ8Block;
    parallel_for((rows + kRows - 1) / kRows, [&](std::size_t block) {
        for (std::size_t r = block * kRows; r < rows && r < (block + 1) * kRows; ++r) {
            quantize_q8_0(x + r * in_features, xq + r * in_features, xs + r * blocks, in_features);
        }
    });
}

// Rows are processed in blocks of kQuantRows so a block of activations stays in cache while it meets every panel.
constexpr std::size_t kQuantRows = 64;

// out[rows, out_features] = Q8_0 linear of x[rows, in_features] (in_features a multiple of 32) with weights
// wq[out_features, in_features] and scales ws[out_features, in_features / 32].
inline void linear_q8(const float* x, const std::int8_t* wq, const float* ws, const float* bias, float* out,
                      std::size_t rows, std::size_t in_features, std::size_t out_features) {
    const std::size_t blocks = in_features / kQ8Block;
    std::vector<std::int8_t> xq(rows * in_features);
    std::vector<float> xs(rows * blocks);
    quantize_rows_q8_0(x, xq.data(), xs.data(), rows, in_features);
    const LinearQ8PanelFn kernel = linear_q8_panel_kernel();
    const std::size_t panels = (out_features + 15) / 16;
    const std::size_t row_blocks = (rows + kQuantRows - 1) / kQuantRows;
    parallel_for(panels * row_blocks, [&](std::size_t task) {
        const std::size_t p = task % panels;
        const std::size_t r0 = (task / panels) * kQuantRows;
        const std::size_t count = r0 + kQuantRows <= rows ? kQuantRows : rows - r0;
        kernel(xq.data() + r0 * in_features, xs.data() + r0 * blocks, wq + p * 16 * in_features, ws + p * 16 * blocks,
               bias, out + r0 * out_features, count, in_features, out_features, p * 16);
    });
}

// out[rows, out_features] = Q4_0 linear of x[rows, in_features] with packed weights wq[out_features, in_features / 2]
// and scales ws[out_features, in_features / 32]: the bits of linear_q8 with the weights unpacked to int8.
inline void linear_q4(const float* x, const std::uint8_t* wq, const float* ws, const float* bias, float* out,
                      std::size_t rows, std::size_t in_features, std::size_t out_features) {
    const std::size_t blocks = in_features / kQ8Block;
    std::vector<std::int8_t> xq(rows * in_features);
    std::vector<float> xs(rows * blocks);
    quantize_rows_q8_0(x, xq.data(), xs.data(), rows, in_features);
    const LinearQ4PanelFn kernel = linear_q4_panel_kernel();
    const std::size_t panels = (out_features + 15) / 16;
    const std::size_t row_blocks = (rows + kQuantRows - 1) / kQuantRows;
    parallel_for(panels * row_blocks, [&](std::size_t task) {
        const std::size_t p = task % panels;
        const std::size_t r0 = (task / panels) * kQuantRows;
        const std::size_t count = r0 + kQuantRows <= rows ? kQuantRows : rows - r0;
        kernel(xq.data() + r0 * in_features, xs.data() + r0 * blocks, wq + p * 16 * (in_features / 2),
               ws + p * 16 * blocks, bias, out + r0 * out_features, count, in_features, out_features, p * 16);
    });
}

}  // namespace dllm
