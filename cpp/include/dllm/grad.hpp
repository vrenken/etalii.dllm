// Backward kernels (reverse-mode gradients) for the decoder, a cross-entropy loss and the AdamW update.
//
// The rules of nn.hpp carry over: every gradient element has one accumulation, in double, over the reduced index in
// ascending order, rounded to float once at the end. Where a gradient sums over positions (weight gradients, key and
// value gradients, embedding rows) the positions are visited in ascending order, then heads in ascending order,
// so the bits depend only on the inputs, never on scheduling. See docs/kernels.md and docs/training.md.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "dllm/math.hpp"
#include "dllm/nn.hpp"
#include "dllm/parallel.hpp"

namespace dllm {

// Gradients of out = x @ w^T + bias, with x [rows, in], w [out, in] and dy [rows, out].
//   dx[r, k] = sum_n dy[r, n] w[n, k]   (n ascending)
//   dw[n, k] = sum_r dy[r, n] x[r, k]   (r ascending)
//   db[n]    = sum_r dy[r, n]           (r ascending)
// dx, dw and db may each be null when that gradient is not needed.
inline void linear_backward(const float* x, const float* w, const float* dy, float* dx, float* dw, float* db,
                            std::size_t rows, std::size_t in_features, std::size_t out_features) {
    // Rows of dx, and rows of dw with their db entry, are independent tasks for the thread pool; each keeps the
    // accumulation order above.
    if (dx != nullptr) {
        parallel_for(rows, [&](std::size_t r) {
            thread_local std::vector<double> acc;
            acc.assign(in_features, 0.0);
            const float* dyr = dy + r * out_features;
            for (std::size_t n = 0; n < out_features; ++n) {
                const double g = dyr[n];
                const float* wn = w + n * in_features;
                for (std::size_t k = 0; k < in_features; ++k) {
                    acc[k] += g * static_cast<double>(wn[k]);
                }
            }
            for (std::size_t k = 0; k < in_features; ++k) {
                dx[r * in_features + k] = static_cast<float>(acc[k]);
            }
        });
    }
    if (dw != nullptr || db != nullptr) {
        parallel_for(out_features, [&](std::size_t n) {
            thread_local std::vector<double> acc;
            acc.assign(in_features, 0.0);
            double bias_acc = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                const double g = dy[r * out_features + n];
                bias_acc += g;
                if (dw != nullptr) {
                    const float* xr = x + r * in_features;
                    for (std::size_t k = 0; k < in_features; ++k) {
                        acc[k] += g * static_cast<double>(xr[k]);
                    }
                }
            }
            if (dw != nullptr) {
                for (std::size_t k = 0; k < in_features; ++k) {
                    dw[n * in_features + k] = static_cast<float>(acc[k]);
                }
            }
            if (db != nullptr) {
                db[n] = static_cast<float>(bias_acc);
            }
        });
    }
}

// Gradients of RMSNorm y = x * inv_rms * g (g = w, or 1 + w in double with add_unit_offset as in Gemma; g = 1 when
// w is null), with inv_rms = 1 / sqrt(mean(x^2) + eps).
//   dx_i = inv_rms (g_i dy_i) - x_i inv_rms^3 / dim * sum_j g_j dy_j x_j   (j ascending)
//   dw_i = sum_r dy[r, i] x[r, i] inv_rms_r                               (r ascending)
// inv_rms is recomputed exactly as the forward kernel computes it.
inline void rms_norm_backward(const float* x, const float* weight, const float* dy, float* dx, float* dw,
                              std::size_t rows, std::size_t dim, double eps, bool add_unit_offset = false) {
    const double offset = add_unit_offset ? 1.0 : 0.0;
    std::vector<double> dw_acc(dim, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * dim;
        const float* dyr = dy + r * dim;
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            sum_sq += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
        }
        const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + eps);
        double dot_acc = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            const double wi = weight != nullptr ? offset + static_cast<double>(weight[i]) : 1.0;
            dot_acc += wi * static_cast<double>(dyr[i]) * static_cast<double>(xr[i]);
            dw_acc[i] += static_cast<double>(dyr[i]) * static_cast<double>(xr[i]) * inv_rms;
        }
        const double coefficient = dot_acc * inv_rms * inv_rms * inv_rms / static_cast<double>(dim);
        if (dx != nullptr) {
            for (std::size_t i = 0; i < dim; ++i) {
                const double wi = weight != nullptr ? offset + static_cast<double>(weight[i]) : 1.0;
                dx[r * dim + i] = static_cast<float>(inv_rms * wi * static_cast<double>(dyr[i]) -
                                                     static_cast<double>(xr[i]) * coefficient);
            }
        }
    }
    if (dw != nullptr) {
        for (std::size_t i = 0; i < dim; ++i) {
            dw[i] = static_cast<float>(dw_acc[i]);
        }
    }
}

// Gradients of LayerNorm y = (x - mean) inv_std w + b (layer_norm in nn.hpp), with xhat_i = (x_i - mean) inv_std,
// mean and inv_std recomputed exactly as the forward kernel computes them, and g_i = w_i dy_i:
//   dx_i = inv_std (g_i - (sum_j g_j) / dim - xhat_i (sum_j g_j xhat_j) / dim)   (j ascending)
//   dw_i = sum_r dy[r, i] xhat[r, i],  db_i = sum_r dy[r, i]                      (r ascending)
// Every value is a double until the one rounding of each output; dx, dw and db may each be null.
inline void layer_norm_backward(const float* x, const float* weight, const float* dy, float* dx, float* dw, float* db,
                                std::size_t rows, std::size_t dim, double eps) {
    std::vector<double> dw_acc(dim, 0.0);
    std::vector<double> db_acc(dim, 0.0);
    std::vector<double> xhat(dim);
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * dim;
        const float* dyr = dy + r * dim;
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
        double g_sum = 0.0;
        double gx_sum = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            xhat[i] = (static_cast<double>(xr[i]) - mean) * inv_std;
            const double g = static_cast<double>(weight[i]) * static_cast<double>(dyr[i]);
            g_sum += g;
            gx_sum += g * xhat[i];
            dw_acc[i] += static_cast<double>(dyr[i]) * xhat[i];
            db_acc[i] += static_cast<double>(dyr[i]);
        }
        if (dx != nullptr) {
            const double g_mean = g_sum / static_cast<double>(dim);
            const double gx_mean = gx_sum / static_cast<double>(dim);
            for (std::size_t i = 0; i < dim; ++i) {
                const double g = static_cast<double>(weight[i]) * static_cast<double>(dyr[i]);
                dx[r * dim + i] = static_cast<float>(inv_std * (g - g_mean - xhat[i] * gx_mean));
            }
        }
    }
    for (std::size_t i = 0; i < dim; ++i) {
        if (dw != nullptr) {
            dw[i] = static_cast<float>(dw_acc[i]);
        }
        if (db != nullptr) {
            db[i] = static_cast<float>(db_acc[i]);
        }
    }
}

// d gelu(x) / dx for the exact GELU 0.5 x erfc(-x / sqrt(2)):
//   0.5 erfc(-x / sqrt(2)) + x exp(-x^2 / 2) / sqrt(2 pi); returns dy times that in double, rounded once.
inline float gelu_backward(float x, float dy) {
    constexpr double inv_sqrt2 = 7.07106781186547524401e-01;
    constexpr double inv_sqrt_2pi = 3.98942280401432677940e-01;
    const double d = x;
    const double slope = 0.5 * dllm::erfc(-d * inv_sqrt2) + d * dllm::exp(-0.5 * d * d) * inv_sqrt_2pi;
    return static_cast<float>(static_cast<double>(dy) * slope);
}

// d silu(x) / dx = s (1 + x (1 - s)) with s = sigmoid(x); returns dy times that, rounded once.
inline float silu_backward(float x, float dy) {
    const double d = x;
    const double s = sigmoid(d);
    return static_cast<float>(static_cast<double>(dy) * s * (1.0 + d * (1.0 - s)));
}

// d gelu_tanh(x) / dx with u = sqrt(2/pi) (x + 0.044715 x^3) and t = tanh(u):
//   0.5 (1 + t) + 0.5 x (1 - t^2) sqrt(2/pi) (1 + 3 * 0.044715 x^2); returns dy times that, rounded once.
inline float gelu_tanh_backward(float x, float dy) {
    constexpr double sqrt_2_over_pi = 7.97884560802865355879e-01;
    const double d = x;
    const double t = dllm::tanh(sqrt_2_over_pi * (d + 0.044715 * d * d * d));
    const double slope = 0.5 * (1.0 + t) + 0.5 * d * (1.0 - t * t) * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * d * d);
    return static_cast<float>(static_cast<double>(dy) * slope);
}

// d softcap(x) / dx = 1 - tanh(x / cap)^2; returns dy times that, rounded once.
inline float softcap_backward(float x, float dy, double cap) {
    const double t = dllm::tanh(static_cast<double>(x) / cap);
    return static_cast<float>(static_cast<double>(dy) * (1.0 - t * t));
}

// Gradients of attention() (nn.hpp) with the same shapes and masking. For every (query t, head h), in t-ascending
// then h-ascending order, the probabilities p are recomputed exactly as the forward kernel does, then
//   dp_j = sum_i dout_i v[j, i]           (i ascending)
//   D    = sum_j p_j dp_j                 (j ascending)
//   ds_j = p_j (dp_j - D)                  (times 1 - tanh(s_j / softcap)^2 when softcap > 0)
//   dq_i = scale sum_j ds_j k[j, i]       (j ascending)
//   dk[j] += scale ds_j q,  dv[j] += p_j dout   (double accumulators, visited in (t, h) order)
inline void attention_backward(const float* q, const float* k, const float* v, const float* dout, float* dq,
                               float* dk, float* dv, std::size_t q_len, std::size_t kv_len, std::size_t q_heads,
                               std::size_t kv_heads, std::size_t head_dim, std::size_t value_dim, double scale,
                               bool causal, std::size_t q_offset, std::size_t window = 0,
                               double softcap = 0.0) {
    if (kv_heads == 0 || q_heads % kv_heads != 0) {
        throw std::invalid_argument("q_heads must be a multiple of kv_heads");
    }
    const std::size_t group = q_heads / kv_heads;
    std::vector<double> probs(kv_len);
    std::vector<double> slopes(kv_len, 1.0);
    std::vector<double> dprobs(kv_len);
    std::vector<double> dq_acc(head_dim);
    std::vector<double> dk_acc(kv_len * kv_heads * head_dim, 0.0);
    std::vector<double> dv_acc(kv_len * kv_heads * value_dim, 0.0);
    for (std::size_t t = 0; t < q_len; ++t) {
        const AttentionSpan span = attention_span(t, kv_len, causal, q_offset, window);
        const std::size_t visible = span.count;
        // Keys first .. first + visible - 1; index j below counts from first.
        const float* kw = k + span.first * kv_heads * head_dim;
        const float* vw = v + span.first * kv_heads * value_dim;
        double* dkw = dk_acc.data() + span.first * kv_heads * head_dim;
        double* dvw = dv_acc.data() + span.first * kv_heads * value_dim;
        for (std::size_t h = 0; h < q_heads; ++h) {
            const std::size_t kvh = h / group;
            const float* qh = q + (t * q_heads + h) * head_dim;
            const float* doh = dout + (t * q_heads + h) * value_dim;
            float* dqh = dq + (t * q_heads + h) * head_dim;
            if (visible == 0) {
                for (std::size_t i = 0; i < head_dim; ++i) {
                    dqh[i] = 0.0f;
                }
                continue;
            }
            double max = 0.0;
            for (std::size_t j = 0; j < visible; ++j) {
                const float* kj = kw + (j * kv_heads + kvh) * head_dim;
                double dotp = 0.0;
                for (std::size_t i = 0; i < head_dim; ++i) {
                    dotp += static_cast<double>(qh[i]) * static_cast<double>(kj[i]);
                }
                probs[j] = dotp * scale;
                if (softcap > 0.0) {
                    const double t = dllm::tanh(probs[j] / softcap);
                    probs[j] = softcap * t;
                    slopes[j] = 1.0 - t * t;
                }
                if (j == 0 || probs[j] > max) {
                    max = probs[j];
                }
            }
            double total = 0.0;
            for (std::size_t j = 0; j < visible; ++j) {
                probs[j] = dllm::exp(probs[j] - max);
                total += probs[j];
            }
            const double inv = 1.0 / total;
            double weighted = 0.0;
            for (std::size_t j = 0; j < visible; ++j) {
                probs[j] *= inv;
                const float* vj = vw + (j * kv_heads + kvh) * value_dim;
                double dotp = 0.0;
                for (std::size_t i = 0; i < value_dim; ++i) {
                    dotp += static_cast<double>(doh[i]) * static_cast<double>(vj[i]);
                }
                dprobs[j] = dotp;
                weighted += probs[j] * dotp;
            }
            for (std::size_t i = 0; i < head_dim; ++i) {
                dq_acc[i] = 0.0;
            }
            for (std::size_t j = 0; j < visible; ++j) {
                const double ds = probs[j] * (dprobs[j] - weighted) * slopes[j] * scale;
                const float* kj = kw + (j * kv_heads + kvh) * head_dim;
                double* dkj = dkw + (j * kv_heads + kvh) * head_dim;
                double* dvj = dvw + (j * kv_heads + kvh) * value_dim;
                for (std::size_t i = 0; i < head_dim; ++i) {
                    dq_acc[i] += ds * static_cast<double>(kj[i]);
                    dkj[i] += ds * static_cast<double>(qh[i]);
                }
                for (std::size_t i = 0; i < value_dim; ++i) {
                    dvj[i] += probs[j] * static_cast<double>(doh[i]);
                }
            }
            for (std::size_t i = 0; i < head_dim; ++i) {
                dqh[i] = static_cast<float>(dq_acc[i]);
            }
        }
    }
    for (std::size_t i = 0; i < dk_acc.size(); ++i) {
        dk[i] = static_cast<float>(dk_acc[i]);
    }
    for (std::size_t i = 0; i < dv_acc.size(); ++i) {
        dv[i] = static_cast<float>(dv_acc[i]);
    }
}

// Gradients of biased_attention() (nn.hpp): (dq, dk, dv, dbias) with the same shapes. For every (query t, head h),
// in t-ascending then h-ascending order, the probabilities p are recomputed exactly as the forward kernel does, then
//   dp_j = sum_i dout_i v[j, i]           (i ascending)
//   D    = sum_j p_j dp_j                 (j ascending)
//   g_j  = p_j (dp_j - D) scale           (dbias[h, t, j] = g_j rounded once)
//   dq_i = sum_j g_j k[j, i]              (j ascending)
//   dk[j] += g_j q,  dv[j] += p_j dout    (double accumulators, visited in (t, h) order)
inline void biased_attention_backward(const float* q, const float* k, const float* v, const float* bias,
                                      const float* dout, float* dq, float* dk, float* dv, float* dbias,
                                      std::size_t q_len, std::size_t kv_len, std::size_t heads, std::size_t head_dim,
                                      std::size_t value_dim, double scale) {
    std::vector<double> probs(kv_len);
    std::vector<double> dprobs(kv_len);
    std::vector<double> dq_acc(head_dim);
    std::vector<double> dk_acc(kv_len * heads * head_dim, 0.0);
    std::vector<double> dv_acc(kv_len * heads * value_dim, 0.0);
    for (std::size_t t = 0; t < q_len; ++t) {
        for (std::size_t h = 0; h < heads; ++h) {
            const float* qh = q + (t * heads + h) * head_dim;
            const float* bh = bias + (h * q_len + t) * kv_len;
            const float* doh = dout + (t * heads + h) * value_dim;
            float* dqh = dq + (t * heads + h) * head_dim;
            float* dbh = dbias + (h * q_len + t) * kv_len;
            if (kv_len == 0) {
                for (std::size_t i = 0; i < head_dim; ++i) {
                    dqh[i] = 0.0f;
                }
                continue;
            }
            double max = 0.0;
            for (std::size_t j = 0; j < kv_len; ++j) {
                const float* kj = k + (j * heads + h) * head_dim;
                double dotp = 0.0;
                for (std::size_t i = 0; i < head_dim; ++i) {
                    dotp += static_cast<double>(qh[i]) * static_cast<double>(kj[i]);
                }
                probs[j] = (dotp + static_cast<double>(bh[j])) * scale;
                if (j == 0 || probs[j] > max) {
                    max = probs[j];
                }
            }
            double total = 0.0;
            for (std::size_t j = 0; j < kv_len; ++j) {
                probs[j] = dllm::exp(probs[j] - max);
                total += probs[j];
            }
            const double inv = 1.0 / total;
            double weighted = 0.0;
            for (std::size_t j = 0; j < kv_len; ++j) {
                probs[j] *= inv;
                const float* vj = v + (j * heads + h) * value_dim;
                double dotp = 0.0;
                for (std::size_t i = 0; i < value_dim; ++i) {
                    dotp += static_cast<double>(doh[i]) * static_cast<double>(vj[i]);
                }
                dprobs[j] = dotp;
                weighted += probs[j] * dotp;
            }
            for (std::size_t i = 0; i < head_dim; ++i) {
                dq_acc[i] = 0.0;
            }
            for (std::size_t j = 0; j < kv_len; ++j) {
                const double ds = probs[j] * (dprobs[j] - weighted) * scale;
                dbh[j] = static_cast<float>(ds);
                const float* kj = k + (j * heads + h) * head_dim;
                double* dkj = dk_acc.data() + (j * heads + h) * head_dim;
                double* dvj = dv_acc.data() + (j * heads + h) * value_dim;
                for (std::size_t i = 0; i < head_dim; ++i) {
                    dq_acc[i] += ds * static_cast<double>(kj[i]);
                    dkj[i] += ds * static_cast<double>(qh[i]);
                }
                for (std::size_t i = 0; i < value_dim; ++i) {
                    dvj[i] += probs[j] * static_cast<double>(doh[i]);
                }
            }
            for (std::size_t i = 0; i < head_dim; ++i) {
                dqh[i] = static_cast<float>(dq_acc[i]);
            }
        }
    }
    for (std::size_t i = 0; i < dk_acc.size(); ++i) {
        dk[i] = static_cast<float>(dk_acc[i]);
    }
    for (std::size_t i = 0; i < dv_acc.size(); ++i) {
        dv[i] = static_cast<float>(dv_acc[i]);
    }
}

// Softmax cross-entropy of logits [rows, vocab] against targets (a negative target ignores that row). Returns the
// summed loss sum_r (logsumexp(l_r) - l_r[target_r]) over rows ascending, in double, and writes
// dlogits = (softmax(l_r) - onehot(target_r)) * scale (zero for ignored rows). logsumexp = max + log(sum_j exp(l_j -
// max)) with j ascending, the same order as softmax().
inline double cross_entropy(const float* logits, const std::int64_t* targets, float* dlogits, std::size_t rows,
                            std::size_t vocab, double scale) {
    std::vector<double> e(vocab);
    double loss = 0.0;
    for (std::size_t r = 0; r < rows; ++r) {
        const float* lr = logits + r * vocab;
        float* dr = dlogits + r * vocab;
        const std::int64_t target = targets[r];
        if (target < 0) {
            for (std::size_t j = 0; j < vocab; ++j) {
                dr[j] = 0.0f;
            }
            continue;
        }
        if (static_cast<std::size_t>(target) >= vocab) {
            throw std::invalid_argument("target out of range");
        }
        const double max = lr[argmax(lr, vocab)];
        double total = 0.0;
        for (std::size_t j = 0; j < vocab; ++j) {
            e[j] = dllm::exp(static_cast<double>(lr[j]) - max);
            total += e[j];
        }
        loss += max + dllm::log(total) - static_cast<double>(lr[target]);
        const double inv = 1.0 / total;
        for (std::size_t j = 0; j < vocab; ++j) {
            const double onehot = static_cast<std::size_t>(target) == j ? 1.0 : 0.0;
            dr[j] = static_cast<float>((e[j] * inv - onehot) * scale);
        }
    }
    return loss;
}

// Gradient of an embedding lookup: out[v, :] = sum of dy[r, :] over the rows r with tokens[r] == v, r ascending,
// in double. Rows of tokens that do not occur are zero. The rows are grouped per token with a counting sort that
// keeps positions in ascending order, so the accumulation order is fixed.
inline void embedding_backward(const float* dy, const std::int64_t* tokens, float* out, std::size_t rows,
                               std::size_t dim, std::size_t vocab) {
    std::vector<std::size_t> start(vocab + 1, 0);
    for (std::size_t r = 0; r < rows; ++r) {
        if (tokens[r] < 0 || static_cast<std::size_t>(tokens[r]) >= vocab) {
            throw std::invalid_argument("token id out of range");
        }
        ++start[static_cast<std::size_t>(tokens[r]) + 1];
    }
    for (std::size_t t = 0; t < vocab; ++t) {
        start[t + 1] += start[t];
    }
    std::vector<std::size_t> order(rows);
    std::vector<std::size_t> fill(start.begin(), start.end() - 1);
    for (std::size_t r = 0; r < rows; ++r) {
        order[fill[static_cast<std::size_t>(tokens[r])]++] = r;
    }
    std::vector<double> acc(dim);
    for (std::size_t t = 0; t < vocab; ++t) {
        float* ot = out + t * dim;
        for (std::size_t i = 0; i < dim; ++i) {
            acc[i] = 0.0;
        }
        for (std::size_t s = start[t]; s < start[t + 1]; ++s) {
            const float* dyr = dy + order[s] * dim;
            for (std::size_t i = 0; i < dim; ++i) {
                acc[i] += static_cast<double>(dyr[i]);
            }
        }
        for (std::size_t i = 0; i < dim; ++i) {
            ot[i] = static_cast<float>(acc[i]);
        }
    }
}

// Sum of squares in index order with a double accumulator (for global gradient norms).
inline double sum_squares(const float* values, std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        acc += static_cast<double>(values[i]) * static_cast<double>(values[i]);
    }
    return acc;
}

// One AdamW step (decoupled weight decay, as torch.optim.AdamW), in place, element by element in double:
//   g = grad * grad_scale
//   m = float(beta1 m + (1 - beta1) g);  v = float(beta2 v + (1 - beta2) g^2)
//   param = float(param (1 - lr wd) - lr (m / bc1) / (sqrt(v / bc2) + eps))
// with bc1 = 1 - beta1^step and bc2 = 1 - beta2^step supplied by the caller. The moments are rounded to float
// before they are used, so a run resumed from a checkpoint (which stores the float moments) continues bit for bit.
inline void adamw_step(float* param, const float* grad, float* m, float* v, std::size_t n, double lr, double beta1,
                       double beta2, double eps, double weight_decay, double bias_correction1,
                       double bias_correction2, double grad_scale) {
    const double decay = 1.0 - lr * weight_decay;
    for (std::size_t i = 0; i < n; ++i) {
        const double g = static_cast<double>(grad[i]) * grad_scale;
        const float m_new = static_cast<float>(beta1 * static_cast<double>(m[i]) + (1.0 - beta1) * g);
        const float v_new = static_cast<float>(beta2 * static_cast<double>(v[i]) + (1.0 - beta2) * g * g);
        m[i] = m_new;
        v[i] = v_new;
        const double m_hat = static_cast<double>(m_new) / bias_correction1;
        const double v_hat = static_cast<double>(v_new) / bias_correction2;
        param[i] = static_cast<float>(static_cast<double>(param[i]) * decay - lr * m_hat / (std::sqrt(v_hat) + eps));
    }
}

// Gradient of the router logits [rows, experts] of moe_route() (nn.hpp) from the gradients of its weights dweights
// [rows, k] (rank order, for the experts in indices), plus an optional extra gradient dprobabilities [rows, experts]
// of the softmax probabilities themselves (the load-balancing loss; null when there is none). The top-k choice has no
// gradient. Per row, in row order, the probabilities p are recomputed exactly as the forward kernel does, then in
// double
//   S     = sum_j p[i_j]               (j ascending, rank order; normalize only)
//   C     = sum_j dw_j p[i_j]          (j ascending; normalize only)
//   dp_e  = dprobabilities_e + (dw_j / S - C / S^2  if e = i_j, normalized;  dw_j  if e = i_j;  0 otherwise)
//   D     = sum_e p_e dp_e             (e ascending)
//   dl_e  = float(p_e (dp_e - D))
inline void moe_route_backward(const float* logits, const std::int64_t* indices, const float* dweights,
                               const float* dprobabilities, float* dlogits, std::size_t rows, std::size_t experts,
                               std::size_t k, bool normalize) {
    std::vector<double> scratch(experts);
    std::vector<float> probabilities(experts);
    std::vector<double> dp(experts);
    for (std::size_t r = 0; r < rows; ++r) {
        softmax(logits + r * experts, probabilities.data(), scratch.data(), experts);
        for (std::size_t e = 0; e < experts; ++e) {
            dp[e] = dprobabilities != nullptr ? static_cast<double>(dprobabilities[r * experts + e]) : 0.0;
        }
        double total = 0.0;
        double weighted = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            const std::int64_t e = indices[r * k + j];
            if (e < 0 || static_cast<std::size_t>(e) >= experts) {
                throw std::invalid_argument("expert index out of range");
            }
            total += static_cast<double>(probabilities[e]);
            weighted += static_cast<double>(dweights[r * k + j]) * static_cast<double>(probabilities[e]);
        }
        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t e = static_cast<std::size_t>(indices[r * k + j]);
            const double dw = dweights[r * k + j];
            dp[e] += normalize ? dw / total - weighted / (total * total) : dw;
        }
        double dot = 0.0;
        for (std::size_t e = 0; e < experts; ++e) {
            dot += static_cast<double>(probabilities[e]) * dp[e];
        }
        for (std::size_t e = 0; e < experts; ++e) {
            dlogits[r * experts + e] = static_cast<float>(static_cast<double>(probabilities[e]) * (dp[e] - dot));
        }
    }
}

}  // namespace dllm
