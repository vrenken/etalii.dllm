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

namespace dllm {

// Tile sizes for the matmul loops. Changing them changes speed only: the per-output accumulation order is fixed.
constexpr std::size_t kTileRows = 16;
constexpr std::size_t kTileCols = 64;
constexpr std::size_t kTileDepth = 256;

// out[m, n] = sum_k x[m, k] * w[n, k] (+ bias[n]): a linear layer with weights stored [out_features, in_features].
// Accumulation runs over k ascending in double; the bias is added to the double sum before the final rounding.
inline void linear(const float* x, const float* w, const float* bias, float* out, std::size_t rows,
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

// Scaled dot-product attention with grouped-query heads.
//   q   [q_len, q_heads, head_dim]      k [kv_len, kv_heads, head_dim]      v [kv_len, kv_heads, value_dim]
//   out [q_len, q_heads, value_dim]
// Query head h reads key/value head h / (q_heads / kv_heads). With causal masking, query t sits at absolute
// position q_offset + t and sees keys 0 .. q_offset + t. Each (query, head) row is computed on its own:
// scores in double (dot over head_dim ascending, times scale), softmax with the maximum subtracted and the sum
// over keys ascending, then the value sum over keys ascending. A row's bits therefore do not depend on q_len,
// so a prefill and token-by-token decoding against a KV cache give identical outputs.
inline void attention(const float* q, const float* k, const float* v, float* out, std::size_t q_len,
                      std::size_t kv_len, std::size_t q_heads, std::size_t kv_heads, std::size_t head_dim,
                      std::size_t value_dim, double scale, bool causal, std::size_t q_offset) {
    if (kv_heads == 0 || q_heads % kv_heads != 0) {
        throw std::invalid_argument("q_heads must be a multiple of kv_heads");
    }
    const std::size_t group = q_heads / kv_heads;
    std::vector<double> scores(kv_len);
    std::vector<double> acc(value_dim);
    for (std::size_t t = 0; t < q_len; ++t) {
        std::size_t visible = kv_len;
        if (causal) {
            const std::size_t last = q_offset + t + 1;
            visible = last < kv_len ? last : kv_len;
        }
        for (std::size_t h = 0; h < q_heads; ++h) {
            const std::size_t kvh = h / group;
            const float* qh = q + (t * q_heads + h) * head_dim;
            float* oh = out + (t * q_heads + h) * value_dim;
            if (visible == 0) {
                for (std::size_t i = 0; i < value_dim; ++i) {
                    oh[i] = 0.0f;
                }
                continue;
            }
            double max = 0.0;
            for (std::size_t j = 0; j < visible; ++j) {
                const float* kj = k + (j * kv_heads + kvh) * head_dim;
                double dotp = 0.0;
                for (std::size_t i = 0; i < head_dim; ++i) {
                    dotp += static_cast<double>(qh[i]) * static_cast<double>(kj[i]);
                }
                scores[j] = dotp * scale;
                if (j == 0 || scores[j] > max) {
                    max = scores[j];
                }
            }
            double total = 0.0;
            for (std::size_t j = 0; j < visible; ++j) {
                scores[j] = dllm::exp(scores[j] - max);
                total += scores[j];
            }
            for (std::size_t i = 0; i < value_dim; ++i) {
                acc[i] = 0.0;
            }
            for (std::size_t j = 0; j < visible; ++j) {
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
    }
}

}  // namespace dllm
