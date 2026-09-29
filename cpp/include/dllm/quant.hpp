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

// Rows [0, rows) of the outputs [n0, n0 + 16) of a Q8_0 linear layer, with the given exact block dot product.
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
            const std::int8_t* wn = wq + n * in_features;
            const float* wsn = ws + n * blocks;
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
// cannot saturate; x never holds -128 because quantize_q8_0 clamps to +-127), widened to int32 and reduced.
DLLM_TARGET_AVX2 inline std::int32_t q8_block_dot_avx2(const std::int8_t* xb, const std::int8_t* wb) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb));
    const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb));
    const __m256i pairs = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    const __m256i quads = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(quads), _mm256_extracti128_si256(quads, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4e));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb1));
    return _mm_cvtsi128_si32(sum);
}

DLLM_TARGET_AVX2 inline void linear_q8_panel_avx2(const std::int8_t* xq, const float* xs, const std::int8_t* wq,
                                                  const float* ws, const float* bias, float* out, std::size_t rows,
                                                  std::size_t in_features, std::size_t out_features,
                                                  std::size_t n0) {
    linear_q8_panel<q8_block_dot_avx2>(xq, xs, wq, ws, bias, out, rows, in_features, out_features, n0);
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

// out[rows, out_features] = Q8_0 linear of x[rows, in_features] (in_features a multiple of 32) with weights
// wq[out_features, in_features] and scales ws[out_features, in_features / 32].
inline void linear_q8(const float* x, const std::int8_t* wq, const float* ws, const float* bias, float* out,
                      std::size_t rows, std::size_t in_features, std::size_t out_features) {
    const std::size_t blocks = in_features / kQ8Block;
    std::vector<std::int8_t> xq(rows * in_features);
    std::vector<float> xs(rows * blocks);
    for (std::size_t r = 0; r < rows; ++r) {
        quantize_q8_0(x + r * in_features, xq.data() + r * in_features, xs.data() + r * blocks, in_features);
    }
    const LinearQ8PanelFn kernel = linear_q8_panel_kernel();
    constexpr std::size_t kRows = 64;
    const std::size_t panels = (out_features + 15) / 16;
    const std::size_t row_blocks = (rows + kRows - 1) / kRows;
    parallel_for(panels * row_blocks, [&](std::size_t task) {
        const std::size_t p = task % panels;
        const std::size_t r0 = (task / panels) * kRows;
        const std::size_t count = r0 + kRows <= rows ? kRows : rows - r0;
        kernel(xq.data() + r0 * in_features, xs.data() + r0 * blocks, wq, ws, bias, out + r0 * out_features, count,
               in_features, out_features, p * 16);
    });
}

}  // namespace dllm
