// Neural network kernels: matmul, RMSNorm, activations, RoPE and attention.
//
// Every output element is produced by its own accumulation in one fixed order (ascending index over the reduced
// dimension, double accumulator, one rounding to float at the end). Loop tiling only changes the order in which
// outputs are visited, never the order inside an accumulation, so the bits of a row do not depend on how many other
// rows are computed with it (batch invariance) or on the tile sizes. See docs/kernels.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "dllm/math.hpp"
#include "dllm/parallel.hpp"
#include "dllm/simd.hpp"

namespace dllm {

// Tile sizes for the matmul loops. Changing them changes speed only: the per-output accumulation order is fixed.
constexpr std::size_t kTileRows = 16;
constexpr std::size_t kTileCols = 64;
constexpr std::size_t kTileDepth = 256;

// out[m, n] = sum_k x[m, k] * w[n, k] (+ bias[n]): a linear layer with weights stored [out_features, in_features].
// Accumulation runs over k ascending in double; the bias is added to the double sum before the final rounding.
// This is the plain statement of the order; linear() below computes exactly the same bits, faster.
inline void linear_reference(const float* x, const float* w, const float* bias, float* out, std::size_t rows,
                   std::size_t in_features, std::size_t out_features) {
    for (std::size_t r0 = 0; r0 < rows; r0 += kTileRows) {
        const std::size_t r1 = r0 + kTileRows < rows ? r0 + kTileRows : rows;
        for (std::size_t n0 = 0; n0 < out_features; n0 += kTileCols) {
            const std::size_t n1 = n0 + kTileCols < out_features ? n0 + kTileCols : out_features;
            for (std::size_t r = r0; r < r1; ++r) {
                const float* xr = x + r * in_features;
                for (std::size_t n = n0; n < n1; ++n) {
                    const float* wn = w + n * in_features;
                    double acc = 0.0;
                    for (std::size_t k = 0; k < in_features; ++k) {
                        acc += static_cast<double>(xr[k]) * static_cast<double>(wn[k]);
                    }
                    if (bias != nullptr) {
                        acc += bias[n];
                    }
                    out[r * out_features + n] = static_cast<float>(acc);
                }
            }
        }
    }
}

// The fast linear layer. Outputs are grouped in panels of kPanel consecutive output features. A panel's weights are
// repacked to [in_features][kPanel] so that step k of all kPanel accumulations reads one contiguous vector; the
// accumulators of the panel (one per output, lanes of a SIMD register) then advance together, each still summing
// its own products over k ascending in double. Products of two floats are exact in double, so no rounding happens
// before the add and the sequence of roundings per output is exactly that of linear_reference(). Panels are the
// thread pool's tasks; a panel's outputs are only ever written by the task that owns it.
constexpr std::size_t kPanel = 16;

inline std::size_t panel_count(std::size_t out_features) { return (out_features + kPanel - 1) / kPanel; }

// Copies the weights of outputs [n0, n0 + kPanel) into dst[in_features][kPanel], zero-padding past out_features.
inline void pack_panel(const float* w, float* dst, std::size_t in_features, std::size_t out_features,
                       std::size_t n0) {
    const std::size_t width = n0 + kPanel <= out_features ? kPanel : out_features - n0;
    for (std::size_t j = 0; j < kPanel; ++j) {
        if (j >= width) {
            for (std::size_t k = 0; k < in_features; ++k) {
                dst[k * kPanel + j] = 0.0f;
            }
            continue;
        }
        const float* wn = w + (n0 + j) * in_features;
        for (std::size_t k = 0; k < in_features; ++k) {
            dst[k * kPanel + j] = wn[k];
        }
    }
}

// Packs a whole weight matrix: dst[panel_count(out_features)][in_features][kPanel].
inline void pack_linear(const float* w, float* dst, std::size_t in_features, std::size_t out_features) {
    for (std::size_t p = 0; p < panel_count(out_features); ++p) {
        pack_panel(w, dst + p * in_features * kPanel, in_features, out_features, p * kPanel);
    }
}

// All rows of one packed panel, one row at a time: kPanel double accumulators that the compiler keeps in vector
// registers (SSE2 on x86-64, NEON on arm64), advancing together over k.
DLLM_ALWAYS_INLINE void linear_panel(const float* x, const float* panel, const float* bias, float* out,
                                     std::size_t rows, std::size_t in_features, std::size_t out_features,
                                     std::size_t n0) {
    const std::size_t width = n0 + kPanel <= out_features ? kPanel : out_features - n0;
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * in_features;
        double acc[kPanel] = {};
        for (std::size_t k = 0; k < in_features; ++k) {
            const float* pk = panel + k * kPanel;
            const double v = xr[k];
            for (std::size_t j = 0; j < kPanel; ++j) {
                acc[j] += v * static_cast<double>(pk[j]);
            }
        }
        for (std::size_t j = 0; j < width; ++j) {
            if (bias != nullptr) {
                acc[j] += bias[n0 + j];
            }
            out[r * out_features + n0 + j] = static_cast<float>(acc[j]);
        }
    }
}

// The baseline panel: SSE2 on x86-64 (every x86-64 CPU has it), NEON on arm64, the plain loop elsewhere.
inline void linear_panel_portable(const float* x, const float* panel, const float* bias, float* out,
                                  std::size_t rows, std::size_t in_features, std::size_t out_features,
                                  std::size_t n0) {
#if defined(DLLM_SSE2)
    const std::size_t width = n0 + kPanel <= out_features ? kPanel : out_features - n0;
    alignas(16) double result[kPanel];
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * in_features;
        __m128d acc[8];
        for (auto& a : acc) {
            a = _mm_setzero_pd();
        }
        for (std::size_t k = 0; k < in_features; ++k) {
            const float* pk = panel + k * kPanel;
            const __m128d v = _mm_set1_pd(static_cast<double>(xr[k]));
            for (std::size_t q = 0; q < 4; ++q) {
                const __m128 w = _mm_loadu_ps(pk + 4 * q);
                acc[2 * q] = _mm_add_pd(acc[2 * q], _mm_mul_pd(v, _mm_cvtps_pd(w)));
                acc[2 * q + 1] = _mm_add_pd(acc[2 * q + 1], _mm_mul_pd(v, _mm_cvtps_pd(_mm_movehl_ps(w, w))));
            }
        }
        for (std::size_t q = 0; q < 8; ++q) {
            _mm_store_pd(result + 2 * q, acc[q]);
        }
        for (std::size_t j = 0; j < width; ++j) {
            double value = result[j];
            if (bias != nullptr) {
                value += bias[n0 + j];
            }
            out[r * out_features + n0 + j] = static_cast<float>(value);
        }
    }
#elif defined(DLLM_NEON)
    // vfmaq_f64 is exact-product fused multiply-add, which equals multiply-then-add here (see the x86 panels).
    const std::size_t width = n0 + kPanel <= out_features ? kPanel : out_features - n0;
    alignas(16) double result[2][kPanel];
    std::size_t r = 0;
    while (r < rows) {
        const std::size_t count = r + 2 <= rows ? 2 : 1;
        const float* x0 = x + r * in_features;
        const float* x1 = count == 2 ? x0 + in_features : x0;
        float64x2_t a0[8];
        float64x2_t a1[8];
        for (std::size_t q = 0; q < 8; ++q) {
            a0[q] = vdupq_n_f64(0.0);
            a1[q] = vdupq_n_f64(0.0);
        }
        for (std::size_t k = 0; k < in_features; ++k) {
            const float* pk = panel + k * kPanel;
            const float64x2_t v0 = vdupq_n_f64(static_cast<double>(x0[k]));
            const float64x2_t v1 = vdupq_n_f64(static_cast<double>(x1[k]));
            for (std::size_t q = 0; q < 4; ++q) {
                const float32x4_t w = vld1q_f32(pk + 4 * q);
                const float64x2_t wl = vcvt_f64_f32(vget_low_f32(w));
                const float64x2_t wh = vcvt_high_f64_f32(w);
                a0[2 * q] = vfmaq_f64(a0[2 * q], v0, wl);
                a0[2 * q + 1] = vfmaq_f64(a0[2 * q + 1], v0, wh);
                a1[2 * q] = vfmaq_f64(a1[2 * q], v1, wl);
                a1[2 * q + 1] = vfmaq_f64(a1[2 * q + 1], v1, wh);
            }
        }
        for (std::size_t q = 0; q < 8; ++q) {
            vst1q_f64(result[0] + 2 * q, a0[q]);
            vst1q_f64(result[1] + 2 * q, a1[q]);
        }
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t j = 0; j < width; ++j) {
                double value = result[i][j];
                if (bias != nullptr) {
                    value += bias[n0 + j];
                }
                out[(r + i) * out_features + n0 + j] = static_cast<float>(value);
            }
        }
        r += count;
    }
#else
    linear_panel(x, panel, bias, out, rows, in_features, out_features, n0);
#endif
}

#ifdef DLLM_X86_DISPATCH
// The x86 panels in intrinsics. They use fused multiply-add: a product of two floats is exact in double, so
// fma(x, w, acc) rounds exactly once, like the separate multiply (exact) and add of linear_reference().
DLLM_TARGET_AVX2 inline void linear_panel_avx2(const float* x, const float* panel, const float* bias, float* out,
                                               std::size_t rows, std::size_t in_features, std::size_t out_features,
                                               std::size_t n0) {
    const std::size_t width = n0 + kPanel <= out_features ? kPanel : out_features - n0;
    alignas(32) double result[2][kPanel];
    std::size_t r = 0;
    while (r < rows) {
        const std::size_t count = r + 2 <= rows ? 2 : 1;
        const float* x0 = x + r * in_features;
        const float* x1 = count == 2 ? x0 + in_features : x0;
        __m256d a00 = _mm256_setzero_pd(), a01 = _mm256_setzero_pd(), a02 = _mm256_setzero_pd(),
                a03 = _mm256_setzero_pd();
        __m256d a10 = _mm256_setzero_pd(), a11 = _mm256_setzero_pd(), a12 = _mm256_setzero_pd(),
                a13 = _mm256_setzero_pd();
        for (std::size_t k = 0; k < in_features; ++k) {
            const float* pk = panel + k * kPanel;
            const __m256 lo = _mm256_loadu_ps(pk);
            const __m256 hi = _mm256_loadu_ps(pk + 8);
            const __m256d w0 = _mm256_cvtps_pd(_mm256_castps256_ps128(lo));
            const __m256d w1 = _mm256_cvtps_pd(_mm256_extractf128_ps(lo, 1));
            const __m256d w2 = _mm256_cvtps_pd(_mm256_castps256_ps128(hi));
            const __m256d w3 = _mm256_cvtps_pd(_mm256_extractf128_ps(hi, 1));
            const __m256d v0 = _mm256_set1_pd(static_cast<double>(x0[k]));
            const __m256d v1 = _mm256_set1_pd(static_cast<double>(x1[k]));
            a00 = _mm256_fmadd_pd(v0, w0, a00);
            a01 = _mm256_fmadd_pd(v0, w1, a01);
            a02 = _mm256_fmadd_pd(v0, w2, a02);
            a03 = _mm256_fmadd_pd(v0, w3, a03);
            a10 = _mm256_fmadd_pd(v1, w0, a10);
            a11 = _mm256_fmadd_pd(v1, w1, a11);
            a12 = _mm256_fmadd_pd(v1, w2, a12);
            a13 = _mm256_fmadd_pd(v1, w3, a13);
        }
        _mm256_store_pd(result[0], a00);
        _mm256_store_pd(result[0] + 4, a01);
        _mm256_store_pd(result[0] + 8, a02);
        _mm256_store_pd(result[0] + 12, a03);
        _mm256_store_pd(result[1], a10);
        _mm256_store_pd(result[1] + 4, a11);
        _mm256_store_pd(result[1] + 8, a12);
        _mm256_store_pd(result[1] + 12, a13);
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t j = 0; j < width; ++j) {
                double acc = result[i][j];
                if (bias != nullptr) {
                    acc += bias[n0 + j];
                }
                out[(r + i) * out_features + n0 + j] = static_cast<float>(acc);
            }
        }
        r += count;
    }
}

#endif

using LinearPanelFn = void (*)(const float*, const float*, const float*, float*, std::size_t, std::size_t,
                               std::size_t, std::size_t);

inline LinearPanelFn linear_panel_kernel() {
#ifdef DLLM_X86_DISPATCH
    switch (active_isa()) {
        case Isa::avx2:
            return linear_panel_avx2;
        default:
            break;
    }
#endif
    return linear_panel_portable;
}

// Rows are processed in blocks so a block of x stays in cache while it meets every panel.
constexpr std::size_t kRowBlock = 64;

// linear() with weights already packed by pack_linear(). Bit-identical to linear_reference().
inline void linear_packed(const float* x, const float* packed, const float* bias, float* out, std::size_t rows,
                          std::size_t in_features, std::size_t out_features) {
    const LinearPanelFn kernel = linear_panel_kernel();
    const std::size_t panels = panel_count(out_features);
    const std::size_t row_blocks = (rows + kRowBlock - 1) / kRowBlock;
    parallel_for(panels * row_blocks, [&](std::size_t task) {
        const std::size_t p = task % panels;
        const std::size_t r0 = (task / panels) * kRowBlock;
        const std::size_t count = r0 + kRowBlock <= rows ? kRowBlock : rows - r0;
        kernel(x + r0 * in_features, packed + p * in_features * kPanel, bias, out + r0 * out_features, count,
               in_features, out_features, p * kPanel);
    });
}

// out[m, n] = sum_k x[m, k] * w[n, k] (+ bias[n]) with w [out_features, in_features]; bit-identical to
// linear_reference(). Each task packs its panel into a per-thread buffer first.
inline void linear(const float* x, const float* w, const float* bias, float* out, std::size_t rows,
                   std::size_t in_features, std::size_t out_features) {
    const LinearPanelFn kernel = linear_panel_kernel();
    parallel_for(panel_count(out_features), [&](std::size_t p) {
        thread_local std::vector<float> buffer;
        buffer.resize(in_features * kPanel);
        pack_panel(w, buffer.data(), in_features, out_features, p * kPanel);
        kernel(x, buffer.data(), bias, out, rows, in_features, out_features, p * kPanel);
    });
}

// out[m, n] = sum_k a[m, k] * b[k, n]. Tiled over rows, columns and depth; each output keeps a double accumulator
// across depth tiles, so the sum still runs over k ascending and equals dot(a[m, :], b[:, n]) bit for bit.
inline void matmul(const float* a, const float* b, float* out, std::size_t m, std::size_t k, std::size_t n) {
    std::vector<double> acc(kTileCols);
    for (std::size_t r = 0; r < m; ++r) {
        const float* ar = a + r * k;
        for (std::size_t c0 = 0; c0 < n; c0 += kTileCols) {
            const std::size_t c1 = c0 + kTileCols < n ? c0 + kTileCols : n;
            for (std::size_t c = c0; c < c1; ++c) {
                acc[c - c0] = 0.0;
            }
            for (std::size_t d0 = 0; d0 < k; d0 += kTileDepth) {
                const std::size_t d1 = d0 + kTileDepth < k ? d0 + kTileDepth : k;
                for (std::size_t d = d0; d < d1; ++d) {
                    const double av = ar[d];
                    const float* bd = b + d * n;
                    for (std::size_t c = c0; c < c1; ++c) {
                        acc[c - c0] += av * static_cast<double>(bd[c]);
                    }
                }
            }
            for (std::size_t c = c0; c < c1; ++c) {
                out[r * n + c] = static_cast<float>(acc[c - c0]);
            }
        }
    }
}

// RMSNorm over the last dimension: out = x / sqrt(mean(x^2) + eps) * weight (weight may be null).
// With add_unit_offset the scale is (1 + weight), as in Gemma checkpoints.
inline void rms_norm(const float* x, const float* weight, float* out, std::size_t rows, std::size_t dim, double eps,
                     bool add_unit_offset) {
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * dim;
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            sum_sq += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
        }
        const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + eps);
        for (std::size_t i = 0; i < dim; ++i) {
            double scale = 1.0;
            if (weight != nullptr) {
                scale = add_unit_offset ? 1.0 + static_cast<double>(weight[i]) : static_cast<double>(weight[i]);
            }
            out[r * dim + i] = static_cast<float>(static_cast<double>(xr[i]) * inv_rms * scale);
        }
    }
}

// Elementwise work is split into fixed chunks of kElementChunk values, one thread-pool task each. Every output is
// computed from its own inputs only, so the split never changes a bit; it is fixed so the tasks do not depend on
// the thread count either.
constexpr std::size_t kElementChunk = 8192;

template <typename Fn>
inline void parallel_elementwise(std::size_t n, Fn fn) {
    const std::size_t chunks = (n + kElementChunk - 1) / kElementChunk;
    if (chunks <= 1) {
        for (std::size_t i = 0; i < n; ++i) {
            fn(i);
        }
        return;
    }
    parallel_for(chunks, [&](std::size_t c) {
        const std::size_t end = (c + 1) * kElementChunk < n ? (c + 1) * kElementChunk : n;
        for (std::size_t i = c * kElementChunk; i < end; ++i) {
            fn(i);
        }
    });
}

// Logit soft-capping (Gemma 2): cap * tanh(x / cap), in double, rounded once.
inline void softcap(const float* x, float* out, std::size_t n, double cap) {
    parallel_elementwise(n, [&](std::size_t i) {
        out[i] = static_cast<float>(cap * dllm::tanh(static_cast<double>(x[i]) / cap));
    });
}

// SiLU (swish): x * sigmoid(x), in double, rounded once.
inline float silu(float x) {
    const double d = x;
    return static_cast<float>(d * sigmoid(d));
}

// Exact GELU: 0.5 x (1 + erf(x / sqrt(2))), evaluated as 0.5 x erfc(-x / sqrt(2)) so the tiny negative tail keeps
// its relative accuracy instead of cancelling to zero.
inline float gelu(float x) {
    constexpr double inv_sqrt2 = 7.07106781186547524401e-01;
    const double d = x;
    return static_cast<float>(0.5 * d * dllm::erfc(-d * inv_sqrt2));
}

// GELU with the tanh approximation (GPT-2, Gemma): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
inline float gelu_tanh(float x) {
    constexpr double sqrt_2_over_pi = 7.97884560802865355879e-01;
    const double d = x;
    return static_cast<float>(0.5 * d * (1.0 + dllm::tanh(sqrt_2_over_pi * (d + 0.044715 * d * d * d))));
}

// The gated MLP activation act(gate) * up: the activation rounded to float first, then a float multiply, exactly as
// computing the two steps separately (and as the CUDA swiglu kernel). Act is silu or gelu_tanh.
template <float (*Act)(float)>
inline void gated_activation(const float* gate, const float* up, float* out, std::size_t n) {
    parallel_elementwise(n, [&](std::size_t i) { out[i] = Act(gate[i]) * up[i]; });
}

// Rotary position embedding, in place-free form. x is [tokens, heads, head_dim]; positions[t] is token t's absolute
// position; inv_freq has rotary_dim / 2 entries (rotary_dim <= head_dim, trailing dims pass through unchanged).
// Pair i rotates by angle positions[t] * inv_freq[i] (computed in double with dllm::sin / dllm::cos):
//  - interleaved = false (Hugging Face "rotate_half" layout): pairs (i, i + rotary_dim / 2);
//  - interleaved = true (Meta / GGUF layout): pairs (2i, 2i + 1).
// With inverse = true every pair rotates by the negated angle (sin negated exactly): the transpose of the rotation,
// which is the RoPE backward pass.
inline void rope(const float* x, const std::int64_t* positions, const double* inv_freq, float* out,
                 std::size_t tokens, std::size_t heads, std::size_t head_dim, std::size_t rotary_dim,
                 bool interleaved, bool inverse = false) {
    const std::size_t half = rotary_dim / 2;
    std::vector<double> cos_table(half);
    std::vector<double> sin_table(half);
    for (std::size_t t = 0; t < tokens; ++t) {
        const double pos = static_cast<double>(positions[t]);
        for (std::size_t i = 0; i < half; ++i) {
            const double angle = pos * inv_freq[i];
            cos_table[i] = dllm::cos(angle);
            sin_table[i] = inverse ? -dllm::sin(angle) : dllm::sin(angle);
        }
        for (std::size_t h = 0; h < heads; ++h) {
            const float* xh = x + (t * heads + h) * head_dim;
            float* oh = out + (t * heads + h) * head_dim;
            for (std::size_t i = 0; i < half; ++i) {
                const std::size_t i0 = interleaved ? 2 * i : i;
                const std::size_t i1 = interleaved ? 2 * i + 1 : i + half;
                const double a = xh[i0];
                const double b = xh[i1];
                oh[i0] = static_cast<float>(a * cos_table[i] - b * sin_table[i]);
                oh[i1] = static_cast<float>(a * sin_table[i] + b * cos_table[i]);
            }
            for (std::size_t i = rotary_dim; i < head_dim; ++i) {
                oh[i] = xh[i];
            }
        }
    }
}

// One (query, head) row of attention(): writes oh[value_dim]. Four keys are scored at a time, each with its own
// accumulator, so the order inside every dot product is unchanged. A positive softcap (Gemma 2) maps each scaled
// score s to softcap * tanh(s / softcap), in double.
DLLM_ALWAYS_INLINE void attention_row(const float* qh, const float* k, const float* v, float* oh, double* scores,
                                      double* acc, std::size_t visible, std::size_t kv_heads, std::size_t kvh,
                                      std::size_t head_dim, std::size_t value_dim, double scale, double softcap) {
    if (visible == 0) {
        for (std::size_t i = 0; i < value_dim; ++i) {
            oh[i] = 0.0f;
        }
        return;
    }
    const std::size_t key_stride = kv_heads * head_dim;
    std::size_t j = 0;
    for (; j + 4 <= visible; j += 4) {
        const float* k0 = k + (j * kv_heads + kvh) * head_dim;
        const float* k1 = k0 + key_stride;
        const float* k2 = k1 + key_stride;
        const float* k3 = k2 + key_stride;
        double d0 = 0.0;
        double d1 = 0.0;
        double d2 = 0.0;
        double d3 = 0.0;
        for (std::size_t i = 0; i < head_dim; ++i) {
            const double qi = qh[i];
            d0 += qi * static_cast<double>(k0[i]);
            d1 += qi * static_cast<double>(k1[i]);
            d2 += qi * static_cast<double>(k2[i]);
            d3 += qi * static_cast<double>(k3[i]);
        }
        scores[j] = d0 * scale;
        scores[j + 1] = d1 * scale;
        scores[j + 2] = d2 * scale;
        scores[j + 3] = d3 * scale;
    }
    for (; j < visible; ++j) {
        const float* kj = k + (j * kv_heads + kvh) * head_dim;
        double dotp = 0.0;
        for (std::size_t i = 0; i < head_dim; ++i) {
            dotp += static_cast<double>(qh[i]) * static_cast<double>(kj[i]);
        }
        scores[j] = dotp * scale;
    }
    if (softcap > 0.0) {
        for (j = 0; j < visible; ++j) {
            scores[j] = softcap * dllm::tanh(scores[j] / softcap);
        }
    }
    double max = scores[0];
    for (j = 1; j < visible; ++j) {
        if (scores[j] > max) {
            max = scores[j];
        }
    }
    double total = 0.0;
    for (j = 0; j < visible; ++j) {
        scores[j] = dllm::exp(scores[j] - max);
        total += scores[j];
    }
    for (std::size_t i = 0; i < value_dim; ++i) {
        acc[i] = 0.0;
    }
    for (j = 0; j < visible; ++j) {
        const float* vj = v + (j * kv_heads + kvh) * value_dim;
        const double p = scores[j];
        for (std::size_t i = 0; i < value_dim; ++i) {
            acc[i] += p * static_cast<double>(vj[i]);
        }
    }
    const double inv = 1.0 / total;
    for (std::size_t i = 0; i < value_dim; ++i) {
        oh[i] = static_cast<float>(acc[i] * inv);
    }
}

using AttentionRowFn = void (*)(const float*, const float*, const float*, float*, double*, double*, std::size_t,
                                std::size_t, std::size_t, std::size_t, std::size_t, double, double);

inline void attention_row_portable(const float* qh, const float* k, const float* v, float* oh, double* scores,
                                   double* acc, std::size_t visible, std::size_t kv_heads, std::size_t kvh,
                                   std::size_t head_dim, std::size_t value_dim, double scale, double softcap) {
    attention_row(qh, k, v, oh, scores, acc, visible, kv_heads, kvh, head_dim, value_dim, scale, softcap);
}

#ifdef DLLM_X86_DISPATCH
DLLM_TARGET_AVX2 inline void attention_row_avx2(const float* qh, const float* k, const float* v, float* oh,
                                                double* scores, double* acc, std::size_t visible,
                                                std::size_t kv_heads, std::size_t kvh, std::size_t head_dim,
                                                std::size_t value_dim, double scale, double softcap) {
    attention_row(qh, k, v, oh, scores, acc, visible, kv_heads, kvh, head_dim, value_dim, scale, softcap);
}

#endif

inline AttentionRowFn attention_row_kernel() {
#ifdef DLLM_X86_DISPATCH
    switch (active_isa()) {
        case Isa::avx2:
            return attention_row_avx2;
        default:
            break;
    }
#endif
    return attention_row_portable;
}

// The keys query t attends to: first .. first + count - 1 (see attention()).
struct AttentionSpan {
    std::size_t first;
    std::size_t count;
};

inline AttentionSpan attention_span(std::size_t t, std::size_t kv_len, bool causal, std::size_t q_offset,
                                    std::size_t window) {
    std::size_t end = kv_len;
    if (causal) {
        const std::size_t last = q_offset + t + 1;
        end = last < kv_len ? last : kv_len;
    }
    const std::size_t first = window != 0 && end > window ? end - window : 0;
    return {first, end - first};
}

// Scaled dot-product attention with grouped-query heads.
//   q   [q_len, q_heads, head_dim]      k [kv_len, kv_heads, head_dim]      v [kv_len, kv_heads, value_dim]
//   out [q_len, q_heads, value_dim]
// Query head h reads key/value head h / (q_heads / kv_heads). With causal masking, query t sits at absolute
// position q_offset + t and sees keys 0 .. q_offset + t; a non-zero window (sliding-window attention) limits that
// to the last `window` of them, q_offset + t - window + 1 .. q_offset + t. Each (query, head) row is computed on its own:
// scores in double (dot over head_dim ascending, times scale, then soft-capped when softcap > 0), softmax with the maximum subtracted and the sum
// over keys ascending, then the value sum over keys ascending. A row's bits therefore do not depend on q_len,
// so a prefill and token-by-token decoding against a KV cache give identical outputs.
inline void attention(const float* q, const float* k, const float* v, float* out, std::size_t q_len,
                      std::size_t kv_len, std::size_t q_heads, std::size_t kv_heads, std::size_t head_dim,
                      std::size_t value_dim, double scale, bool causal, std::size_t q_offset,
                      std::size_t window = 0, double softcap = 0.0) {
    if (kv_heads == 0 || q_heads % kv_heads != 0) {
        throw std::invalid_argument("q_heads must be a multiple of kv_heads");
    }
    const std::size_t group = q_heads / kv_heads;
    const AttentionRowFn kernel = attention_row_kernel();
    // One task per (query, head) row; the rows are independent.
    parallel_for(q_len * q_heads, [&](std::size_t task) {
        const std::size_t t = task / q_heads;
        const std::size_t h = task % q_heads;
        const AttentionSpan span = attention_span(t, kv_len, causal, q_offset, window);
        thread_local std::vector<double> scores;
        thread_local std::vector<double> acc;
        scores.resize(kv_len > 0 ? kv_len : 1);
        acc.resize(value_dim > 0 ? value_dim : 1);
        kernel(q + (t * q_heads + h) * head_dim, k + span.first * kv_heads * head_dim,
               v + span.first * kv_heads * value_dim, out + (t * q_heads + h) * value_dim, scores.data(), acc.data(),
               span.count, kv_heads, h / group, head_dim, value_dim, scale, softcap);
    });
}

}  // namespace dllm
