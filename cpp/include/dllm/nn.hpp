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

// The panel kernels work on register tiles: kTileRowsMax rows of x by a slice of the panel's outputs advance
// together over k, so every weight vector loaded and converted serves several rows. x arrives converted to double
// (convert_rows below) so a row value is a plain broadcast in the inner loop. The tile height only depends on the
// kernel and on how many rows are left, and it changes speed only: every output keeps its own accumulator over k.
constexpr std::size_t kTileRowsMax = 6;

// Copies x[rows][in_features] to double (exact), the form the panel kernels read.
inline void convert_rows(const float* x, double* dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = x[i];
    }
}

// Rounds the double sums of rows [r, r + R) and outputs [n0 + j0, n0 + j1) of a panel to float, after the bias.
template <std::size_t R, std::size_t W>
DLLM_ALWAYS_INLINE void store_tile(const double (&result)[R][W], const float* bias, float* out, std::size_t r,
                                   std::size_t out_features, std::size_t n0, std::size_t j0, std::size_t j1) {
    for (std::size_t i = 0; i < R; ++i) {
        for (std::size_t j = j0; j < j1; ++j) {
            double acc = result[i][j - j0];
            if (bias != nullptr) {
                acc += bias[n0 + j];
            }
            out[(r + i) * out_features + n0 + j] = static_cast<float>(acc);
        }
    }
}

// Calls tile<R>(r) over rows [0, rows): tiles of kTileRowsMax rows, then one tile of the rows left.
template <template <std::size_t> class Tile, class... Args>
DLLM_ALWAYS_INLINE void for_each_tile(std::size_t rows, Args&&... args) {
    std::size_t r = 0;
    for (; r + kTileRowsMax <= rows; r += kTileRowsMax) {
        Tile<kTileRowsMax>::run(r, args...);
    }
    switch (rows - r) {
        case 5: Tile<5>::run(r, args...); break;
        case 4: Tile<4>::run(r, args...); break;
        case 3: Tile<3>::run(r, args...); break;
        case 2: Tile<2>::run(r, args...); break;
        case 1: Tile<1>::run(r, args...); break;
        default: break;
    }
}

struct PanelArgs {
    const double* x;
    const float* panel;
    const float* bias;
    float* out;
    std::size_t in_features;
    std::size_t out_features;
    std::size_t n0;
    std::size_t width;
};

// The plain tile: R rows by the whole panel, kPanel double accumulators per row.
template <std::size_t R>
struct PlainTile {
    static DLLM_ALWAYS_INLINE void run(std::size_t r, const PanelArgs& a) {
        double acc[R][kPanel] = {};
        for (std::size_t k = 0; k < a.in_features; ++k) {
            const float* pk = a.panel + k * kPanel;
            for (std::size_t i = 0; i < R; ++i) {
                const double v = a.x[(r + i) * a.in_features + k];
                for (std::size_t j = 0; j < kPanel; ++j) {
                    acc[i][j] += v * static_cast<double>(pk[j]);
                }
            }
        }
        store_tile(acc, a.bias, a.out, r, a.out_features, a.n0, 0, a.width);
    }
};

#if defined(DLLM_SSE2)
// SSE2 (every x86-64 CPU): R rows by four outputs, two accumulators of two doubles per row.
template <std::size_t R>
struct Sse2Tile {
    static DLLM_ALWAYS_INLINE void run(std::size_t r, const PanelArgs& a) {
        for (std::size_t h = 0; h < a.width; h += 4) {
            __m128d acc[R][2];
            for (std::size_t i = 0; i < R; ++i) {
                acc[i][0] = _mm_setzero_pd();
                acc[i][1] = _mm_setzero_pd();
            }
            const float* pk = a.panel + h;
            const double* xr = a.x + r * a.in_features;
            for (std::size_t k = 0; k < a.in_features; ++k, pk += kPanel) {
                const __m128 w = _mm_loadu_ps(pk);
                const __m128d w0 = _mm_cvtps_pd(w);
                const __m128d w1 = _mm_cvtps_pd(_mm_movehl_ps(w, w));
                for (std::size_t i = 0; i < R; ++i) {
                    const __m128d v = _mm_set1_pd(xr[i * a.in_features + k]);
                    acc[i][0] = _mm_add_pd(acc[i][0], _mm_mul_pd(v, w0));
                    acc[i][1] = _mm_add_pd(acc[i][1], _mm_mul_pd(v, w1));
                }
            }
            alignas(16) double result[R][4];
            for (std::size_t i = 0; i < R; ++i) {
                _mm_store_pd(result[i], acc[i][0]);
                _mm_store_pd(result[i] + 2, acc[i][1]);
            }
            store_tile(result, a.bias, a.out, r, a.out_features, a.n0, h, h + 4 < a.width ? h + 4 : a.width);
        }
    }
};
#elif defined(DLLM_NEON)
// NEON (every arm64 CPU, 32 vector registers): R rows by eight outputs, four accumulators of two doubles per row.
// vfmaq_f64 is exact-product fused multiply-add, which equals multiply-then-add here (see the AVX2 tile).
template <std::size_t R>
struct NeonTile {
    static DLLM_ALWAYS_INLINE void run(std::size_t r, const PanelArgs& a) {
        for (std::size_t h = 0; h < a.width; h += 8) {
            float64x2_t acc[R][4];
            for (std::size_t i = 0; i < R; ++i) {
                for (std::size_t q = 0; q < 4; ++q) {
                    acc[i][q] = vdupq_n_f64(0.0);
                }
            }
            const float* pk = a.panel + h;
            const double* xr = a.x + r * a.in_features;
            for (std::size_t k = 0; k < a.in_features; ++k, pk += kPanel) {
                const float32x4_t lo = vld1q_f32(pk);
                const float32x4_t hi = vld1q_f32(pk + 4);
                const float64x2_t w0 = vcvt_f64_f32(vget_low_f32(lo));
                const float64x2_t w1 = vcvt_high_f64_f32(lo);
                const float64x2_t w2 = vcvt_f64_f32(vget_low_f32(hi));
                const float64x2_t w3 = vcvt_high_f64_f32(hi);
                for (std::size_t i = 0; i < R; ++i) {
                    const float64x2_t v = vld1q_dup_f64(xr + i * a.in_features + k);
                    acc[i][0] = vfmaq_f64(acc[i][0], v, w0);
                    acc[i][1] = vfmaq_f64(acc[i][1], v, w1);
                    acc[i][2] = vfmaq_f64(acc[i][2], v, w2);
                    acc[i][3] = vfmaq_f64(acc[i][3], v, w3);
                }
            }
            alignas(16) double result[R][8];
            for (std::size_t i = 0; i < R; ++i) {
                for (std::size_t q = 0; q < 4; ++q) {
                    vst1q_f64(result[i] + 2 * q, acc[i][q]);
                }
            }
            store_tile(result, a.bias, a.out, r, a.out_features, a.n0, h, h + 8 < a.width ? h + 8 : a.width);
        }
    }
};
#endif

// The baseline panel: SSE2 on x86-64, NEON on arm64, the plain loop elsewhere.
inline void linear_panel_portable(const double* x, const float* panel, const float* bias, float* out,
                                  std::size_t rows, std::size_t in_features, std::size_t out_features,
                                  std::size_t n0) {
    const PanelArgs args{x, panel, bias, out, in_features, out_features, n0,
                         n0 + kPanel <= out_features ? kPanel : out_features - n0};
#if defined(DLLM_SSE2)
    for_each_tile<Sse2Tile>(rows, args);
#elif defined(DLLM_NEON)
    for_each_tile<NeonTile>(rows, args);
#else
    for_each_tile<PlainTile>(rows, args);
#endif
}

#ifdef DLLM_X86_DISPATCH
// AVX2: R rows by C vectors of four outputs. Tiles of three or more rows take half a panel (C = 2): with R = 6,
// twelve accumulators, two converted weight vectors and one broadcast fit the sixteen vector registers. One or two
// rows take the whole panel (C = 4), so there are enough independent accumulators to hide the FMA latency. It uses
// fused multiply-add: a product of two floats is exact in double, so fma(x, w, acc) rounds exactly once, like the
// separate multiply (exact) and add of linear_reference().
template <std::size_t R>
struct Avx2Tile {
    static constexpr std::size_t C = R <= 2 ? 4 : 2;
    // A call per tile: AVX2 code cannot be forced inline into the generic for_each_tile.
    static DLLM_TARGET_AVX2 void run(std::size_t r, const PanelArgs& a) {
        for (std::size_t h = 0; h < a.width; h += 4 * C) {
            __m256d acc[R][C];
            for (std::size_t i = 0; i < R; ++i) {
                for (std::size_t c = 0; c < C; ++c) {
                    acc[i][c] = _mm256_setzero_pd();
                }
            }
            const float* pk = a.panel + h;
            const double* xr = a.x + r * a.in_features;
            for (std::size_t k = 0; k < a.in_features; ++k, pk += kPanel) {
                __m256d w[C];
                for (std::size_t c = 0; c < C; ++c) {
                    w[c] = _mm256_cvtps_pd(_mm_loadu_ps(pk + 4 * c));
                }
                for (std::size_t i = 0; i < R; ++i) {
                    const __m256d v = _mm256_broadcast_sd(xr + i * a.in_features + k);
                    for (std::size_t c = 0; c < C; ++c) {
                        acc[i][c] = _mm256_fmadd_pd(v, w[c], acc[i][c]);
                    }
                }
            }
            alignas(32) double result[R][4 * C];
            for (std::size_t i = 0; i < R; ++i) {
                for (std::size_t c = 0; c < C; ++c) {
                    _mm256_store_pd(result[i] + 4 * c, acc[i][c]);
                }
            }
            const std::size_t end = h + 4 * C < a.width ? h + 4 * C : a.width;
            store_tile(result, a.bias, a.out, r, a.out_features, a.n0, h, end);
        }
    }
};

DLLM_TARGET_AVX2 inline void linear_panel_avx2(const double* x, const float* panel, const float* bias, float* out,
                                               std::size_t rows, std::size_t in_features, std::size_t out_features,
                                               std::size_t n0) {
    const PanelArgs args{x, panel, bias, out, in_features, out_features, n0,
                         n0 + kPanel <= out_features ? kPanel : out_features - n0};
    for_each_tile<Avx2Tile>(rows, args);
}

#endif

using LinearPanelFn = void (*)(const double*, const float*, const float*, float*, std::size_t, std::size_t,
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

// Rows are processed in blocks so a block of x stays in cache while it meets every panel; a multiple of the tile
// height.
constexpr std::size_t kRowBlock = 4 * kTileRowsMax;

// linear() with weights already packed by pack_linear(). Bit-identical to linear_reference().
inline void linear_packed(const float* x, const float* packed, const float* bias, float* out, std::size_t rows,
                          std::size_t in_features, std::size_t out_features) {
    const LinearPanelFn kernel = linear_panel_kernel();
    const std::size_t panels = panel_count(out_features);
    const std::size_t row_blocks = (rows + kRowBlock - 1) / kRowBlock;
    thread_local std::vector<double> buffer;  // reused by this caller's later calls
    buffer.resize(rows * in_features);
    double* const xd = buffer.data();  // the lambdas below run on other threads, with other thread_locals
    parallel_for(row_blocks, [&](std::size_t block) {
        const std::size_t r0 = block * kRowBlock;
        const std::size_t count = r0 + kRowBlock <= rows ? kRowBlock : rows - r0;
        convert_rows(x + r0 * in_features, xd + r0 * in_features, count * in_features);
    });
    parallel_for(panels * row_blocks, [&](std::size_t task) {
        const std::size_t p = task % panels;
        const std::size_t r0 = (task / panels) * kRowBlock;
        const std::size_t count = r0 + kRowBlock <= rows ? kRowBlock : rows - r0;
        kernel(xd + r0 * in_features, packed + p * in_features * kPanel, bias, out + r0 * out_features, count,
               in_features, out_features, p * kPanel);
    });
}

// out[m, n] = sum_k x[m, k] * w[n, k] (+ bias[n]) with w [out_features, in_features]; bit-identical to
// linear_reference(). Each task packs its panel into a per-thread buffer first.
inline void linear(const float* x, const float* w, const float* bias, float* out, std::size_t rows,
                   std::size_t in_features, std::size_t out_features) {
    const LinearPanelFn kernel = linear_panel_kernel();
    std::vector<double> xd(rows * in_features);
    convert_rows(x, xd.data(), xd.size());
    parallel_for(panel_count(out_features), [&](std::size_t p) {
        thread_local std::vector<float> buffer;
        buffer.resize(in_features * kPanel);
        pack_panel(w, buffer.data(), in_features, out_features, p * kPanel);
        kernel(xd.data(), buffer.data(), bias, out, rows, in_features, out_features, p * kPanel);
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

// LayerNorm over the last dimension (BERT): mean = (sum x_i) / dim and var = (sum (x_i - mean)^2) / dim, both sums
// ascending in double, then out_i = (x_i - mean) * (1 / sqrt(var + eps)) * w_i + b_i in double, rounded once.
inline void layer_norm(const float* x, const float* weight, const float* bias, float* out, std::size_t rows,
                       std::size_t dim, double eps) {
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * dim;
        double total = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            total += static_cast<double>(xr[i]);
        }
        const double mean = total / static_cast<double>(dim);
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            const double centred = static_cast<double>(xr[i]) - mean;
            sum_sq += centred * centred;
        }
        const double inv_std = 1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + eps);
        for (std::size_t i = 0; i < dim; ++i) {
            const double scaled = (static_cast<double>(xr[i]) - mean) * inv_std * static_cast<double>(weight[i]);
            out[r * dim + i] = static_cast<float>(scaled + static_cast<double>(bias[i]));
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

// Logistic sigmoid of a float, in double, rounded once (Qwen2-MoE's shared-expert gate).
inline float sigmoid_float(float x) { return static_cast<float>(sigmoid(static_cast<double>(x))); }

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

// The softmax and value sum of one attention row whose scaled scores[0, visible) are ready: a positive softcap
// (Gemma 2) maps each score s to softcap * tanh(s / softcap), in double; the maximum is subtracted, exponentials
// and their total run over keys ascending, then the value sum over keys ascending, divided by the total once.
DLLM_ALWAYS_INLINE void attention_finish(double* scores, const float* v, float* oh, double* acc, std::size_t visible,
                                         std::size_t kv_heads, std::size_t kvh, std::size_t value_dim,
                                         double softcap) {
    std::size_t j = 0;
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

// One (query, head) row of attention(): writes oh[value_dim]. Four keys are scored at a time, each with its own
// accumulator, so the order inside every dot product is unchanged; attention_finish() does the rest. This is the
// plain statement of a row; attention() computes the same bits with attention_tile().
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
    attention_finish(scores, v, oh, acc, visible, kv_heads, kvh, value_dim, softcap);
}

// Up to kQueryTile rows that read the same key/value head, computed together: lane l of every score accumulator
// belongs to row l, so SIMD registers hold different rows while each lane sums its own dot product over head_dim
// ascending in double, exactly as attention_row(). The rows may see different keys (causal masks, windows): scores
// are computed for the union of their spans and each row's softmax and value sum (attention_finish) use only its
// own span, so a row's bits never depend on the rows it is tiled with.
constexpr std::size_t kQueryTile = 4;

struct AttentionTile {
    const float* q[kQueryTile];
    float* out[kQueryTile];
    std::size_t first[kQueryTile];
    std::size_t count[kQueryTile];
    std::size_t lanes;
};

DLLM_ALWAYS_INLINE void attention_tile(const AttentionTile& tile, const float* k, const float* v, double* scores,
                                       double* acc, float* qt, std::size_t kv_len, std::size_t kv_heads,
                                       std::size_t kvh, std::size_t head_dim, std::size_t value_dim, double scale,
                                       double softcap) {
    std::size_t u0 = kv_len;
    std::size_t u1 = 0;
    for (std::size_t l = 0; l < tile.lanes; ++l) {
        if (tile.count[l] == 0) {
            continue;
        }
        u0 = tile.first[l] < u0 ? tile.first[l] : u0;
        u1 = tile.first[l] + tile.count[l] > u1 ? tile.first[l] + tile.count[l] : u1;
    }
    // Queries transposed to qt[head_dim][kQueryTile]; unused lanes are zero and their scores are never read.
    for (std::size_t i = 0; i < head_dim; ++i) {
        for (std::size_t l = 0; l < kQueryTile; ++l) {
            qt[i * kQueryTile + l] = l < tile.lanes ? tile.q[l][i] : 0.0f;
        }
    }
    const std::size_t key_stride = kv_heads * head_dim;
    std::size_t j = u0;
    for (; j + 4 <= u1; j += 4) {
        const float* k0 = k + (j * kv_heads + kvh) * head_dim;
        double a[4][kQueryTile] = {};
        for (std::size_t i = 0; i < head_dim; ++i) {
            const float* qi = qt + i * kQueryTile;
            for (std::size_t b = 0; b < 4; ++b) {
                const double kb = k0[b * key_stride + i];
                for (std::size_t l = 0; l < kQueryTile; ++l) {
                    a[b][l] += static_cast<double>(qi[l]) * kb;
                }
            }
        }
        for (std::size_t b = 0; b < 4; ++b) {
            for (std::size_t l = 0; l < kQueryTile; ++l) {
                scores[l * kv_len + j + b] = a[b][l] * scale;
            }
        }
    }
    for (; j < u1; ++j) {
        const float* kj = k + (j * kv_heads + kvh) * head_dim;
        double a[kQueryTile] = {};
        for (std::size_t i = 0; i < head_dim; ++i) {
            const double ki = kj[i];
            for (std::size_t l = 0; l < kQueryTile; ++l) {
                a[l] += static_cast<double>(qt[i * kQueryTile + l]) * ki;
            }
        }
        for (std::size_t l = 0; l < kQueryTile; ++l) {
            scores[l * kv_len + j] = a[l] * scale;
        }
    }
    for (std::size_t l = 0; l < tile.lanes; ++l) {
        if (tile.count[l] == 0) {
            for (std::size_t i = 0; i < value_dim; ++i) {
                tile.out[l][i] = 0.0f;
            }
            continue;
        }
        attention_finish(scores + l * kv_len + tile.first[l], v + tile.first[l] * kv_heads * value_dim, tile.out[l],
                         acc, tile.count[l], kv_heads, kvh, value_dim, softcap);
    }
}

using AttentionTileFn = void (*)(const AttentionTile&, const float*, const float*, double*, double*, float*,
                                 std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, double, double);

inline void attention_tile_portable(const AttentionTile& tile, const float* k, const float* v, double* scores,
                                    double* acc, float* qt, std::size_t kv_len, std::size_t kv_heads,
                                    std::size_t kvh, std::size_t head_dim, std::size_t value_dim, double scale,
                                    double softcap) {
    attention_tile(tile, k, v, scores, acc, qt, kv_len, kv_heads, kvh, head_dim, value_dim, scale, softcap);
}

#ifdef DLLM_X86_DISPATCH
DLLM_TARGET_AVX2 inline void attention_tile_avx2(const AttentionTile& tile, const float* k, const float* v,
                                                 double* scores, double* acc, float* qt, std::size_t kv_len,
                                                 std::size_t kv_heads, std::size_t kvh, std::size_t head_dim,
                                                 std::size_t value_dim, double scale, double softcap) {
    attention_tile(tile, k, v, scores, acc, qt, kv_len, kv_heads, kvh, head_dim, value_dim, scale, softcap);
}
#endif

inline AttentionTileFn attention_tile_kernel() {
#ifdef DLLM_X86_DISPATCH
    switch (active_isa()) {
        case Isa::avx2:
            return attention_tile_avx2;
        default:
            break;
    }
#endif
    return attention_tile_portable;
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
    const AttentionTileFn kernel = attention_tile_kernel();
    // The rows of each key/value head, in (query, head) order, cut into tiles of kQueryTile: one task per tile.
    const std::size_t rows = q_len * group;
    const std::size_t tiles = (rows + kQueryTile - 1) / kQueryTile;
    parallel_for(kv_heads * tiles, [&](std::size_t task) {
        const std::size_t kvh = task / tiles;
        const std::size_t r0 = (task % tiles) * kQueryTile;
        AttentionTile tile{};
        tile.lanes = r0 + kQueryTile <= rows ? kQueryTile : rows - r0;
        for (std::size_t l = 0; l < tile.lanes; ++l) {
            const std::size_t t = (r0 + l) / group;
            const std::size_t h = kvh * group + (r0 + l) % group;
            const AttentionSpan span = attention_span(t, kv_len, causal, q_offset, window);
            tile.q[l] = q + (t * q_heads + h) * head_dim;
            tile.out[l] = out + (t * q_heads + h) * value_dim;
            tile.first[l] = span.first;
            tile.count[l] = span.count;
        }
        thread_local std::vector<double> scores;
        thread_local std::vector<double> acc;
        thread_local std::vector<float> qt;
        scores.resize(kQueryTile * (kv_len > 0 ? kv_len : 1));
        acc.resize(value_dim > 0 ? value_dim : 1);
        qt.resize(kQueryTile * (head_dim > 0 ? head_dim : 1));
        kernel(tile, k, v, scores.data(), acc.data(), qt.data(), kv_len, kv_heads, kvh, head_dim, value_dim, scale,
               softcap);
    });
}

// attention() row by row with attention_row(), on one thread: the plain statement of the order, for tests.
inline void attention_reference(const float* q, const float* k, const float* v, float* out, std::size_t q_len,
                                std::size_t kv_len, std::size_t q_heads, std::size_t kv_heads,
                                std::size_t head_dim, std::size_t value_dim, double scale, bool causal,
                                std::size_t q_offset, std::size_t window = 0, double softcap = 0.0) {
    if (kv_heads == 0 || q_heads % kv_heads != 0) {
        throw std::invalid_argument("q_heads must be a multiple of kv_heads");
    }
    const std::size_t group = q_heads / kv_heads;
    std::vector<double> scores(kv_len > 0 ? kv_len : 1);
    std::vector<double> acc(value_dim > 0 ? value_dim : 1);
    for (std::size_t t = 0; t < q_len; ++t) {
        const AttentionSpan span = attention_span(t, kv_len, causal, q_offset, window);
        for (std::size_t h = 0; h < q_heads; ++h) {
            attention_row(q + (t * q_heads + h) * head_dim, k + span.first * kv_heads * head_dim,
                          v + span.first * kv_heads * value_dim, out + (t * q_heads + h) * value_dim, scores.data(),
                          acc.data(), span.count, kv_heads, h / group, head_dim, value_dim, scale, softcap);
        }
    }
}

// Mixture-of-experts routing of rows logits[rows, experts]: per row, the softmax of softmax() (the maximum
// subtracted, exponentials and their total in double over experts ascending, times the reciprocal, rounded to
// float), then the k largest probabilities in a total order (larger first, equal ones by lower expert index). With
// normalize, each chosen probability is divided by their total, summed in double in that rank order, and rounded
// once to float. indices[rows, k] and weights[rows, k] are in rank order. Each row is routed on its own, so the
// result never depends on how many rows are routed together.
inline void moe_route(const float* logits, std::int64_t* indices, float* weights, std::size_t rows,
                      std::size_t experts, std::size_t k, bool normalize) {
    if (k == 0 || k > experts) {
        throw std::invalid_argument("experts per token must be between 1 and the number of experts");
    }
    std::vector<double> scratch(experts);
    std::vector<float> probabilities(experts);
    std::vector<unsigned char> taken(experts);
    for (std::size_t r = 0; r < rows; ++r) {
        softmax(logits + r * experts, probabilities.data(), scratch.data(), experts);
        for (std::size_t e = 0; e < experts; ++e) {
            taken[e] = 0;
        }
        double total = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            std::size_t best = experts;
            for (std::size_t e = 0; e < experts; ++e) {
                if (!taken[e] && (best == experts || probabilities[e] > probabilities[best])) {
                    best = e;
                }
            }
            taken[best] = 1;
            indices[r * k + j] = static_cast<std::int64_t>(best);
            weights[r * k + j] = probabilities[best];
            total += static_cast<double>(probabilities[best]);
        }
        if (normalize) {
            for (std::size_t j = 0; j < k; ++j) {
                weights[r * k + j] = static_cast<float>(static_cast<double>(weights[r * k + j]) / total);
            }
        }
    }
}

}  // namespace dllm
