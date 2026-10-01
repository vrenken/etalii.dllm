// Kernels for interpretability and model editing: attention probabilities, cosine similarities, column means and a
// Cholesky solve.
//
// Like nn.hpp, every output element has one accumulation in a fixed order (ascending index, double accumulator, one
// rounding at the end), threads split outputs and never a reduction, so the results are the same bits on every
// machine, thread count and SIMD path. See docs/kernels.md#interpretability-kernels.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "dllm/math.hpp"
#include "dllm/nn.hpp"
#include "dllm/parallel.hpp"

namespace dllm {

// The attention probabilities attention() uses, written out instead of applied to the values.
//   q [q_len, q_heads, head_dim]   k [kv_len, kv_heads, head_dim]   out [q_len, q_heads, kv_len]
// Scores, soft-capping, masking (causal, window) and the softmax are attention()'s, step for step: each score is a
// double dot product over head_dim ascending times scale, the maximum is subtracted, exp() is dllm::exp and the total
// is summed over the visible keys ascending. Probability j is exp(s_j - max) * (1 / total) rounded once to float;
// keys a query cannot see get 0.
inline void attention_weights(const float* q, const float* k, float* out, std::size_t q_len, std::size_t kv_len,
                              std::size_t q_heads, std::size_t kv_heads, std::size_t head_dim, double scale,
                              bool causal, std::size_t q_offset, std::size_t window = 0, double softcap = 0.0) {
    if (kv_heads == 0 || q_heads % kv_heads != 0) {
        throw std::invalid_argument("q_heads must be a multiple of kv_heads");
    }
    const std::size_t group = q_heads / kv_heads;
    parallel_for(q_len * q_heads, [&](std::size_t task) {
        const std::size_t t = task / q_heads;
        const std::size_t h = task % q_heads;
        const std::size_t kvh = h / group;
        const AttentionSpan span = attention_span(t, kv_len, causal, q_offset, window);
        const float* qh = q + (t * q_heads + h) * head_dim;
        float* row = out + (t * q_heads + h) * kv_len;
        for (std::size_t j = 0; j < kv_len; ++j) {
            row[j] = 0.0f;
        }
        if (span.count == 0) {
            return;
        }
        thread_local std::vector<double> scores;
        scores.resize(span.count);
        for (std::size_t j = 0; j < span.count; ++j) {
            const float* kj = k + ((span.first + j) * kv_heads + kvh) * head_dim;
            double dotp = 0.0;
            for (std::size_t i = 0; i < head_dim; ++i) {
                dotp += static_cast<double>(qh[i]) * static_cast<double>(kj[i]);
            }
            scores[j] = dotp * scale;
        }
        if (softcap > 0.0) {
            for (std::size_t j = 0; j < span.count; ++j) {
                scores[j] = softcap * dllm::tanh(scores[j] / softcap);
            }
        }
        double max = scores[0];
        for (std::size_t j = 1; j < span.count; ++j) {
            if (scores[j] > max) {
                max = scores[j];
            }
        }
        double total = 0.0;
        for (std::size_t j = 0; j < span.count; ++j) {
            scores[j] = dllm::exp(scores[j] - max);
            total += scores[j];
        }
        const double inv = 1.0 / total;
        for (std::size_t j = 0; j < span.count; ++j) {
            row[span.first + j] = static_cast<float>(scores[j] * inv);
        }
    });
}

// out[r] = cos(matrix[r], query) = (matrix[r] . query) / (|matrix[r]| |query|) for matrix [rows, dim]. The dot
// product and both squared norms are double sums over dim ascending; the norms are IEEE square roots. A zero row or
// a zero query gives 0.
inline void cosine_similarity(const float* matrix, const float* query, float* out, std::size_t rows, std::size_t dim) {
    double query_squares = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        query_squares += static_cast<double>(query[i]) * static_cast<double>(query[i]);
    }
    const double query_norm = std::sqrt(query_squares);
    parallel_for(rows, [&](std::size_t r) {
        const float* row = matrix + r * dim;
        double dotp = 0.0;
        double squares = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            dotp += static_cast<double>(row[i]) * static_cast<double>(query[i]);
            squares += static_cast<double>(row[i]) * static_cast<double>(row[i]);
        }
        const double norm = std::sqrt(squares) * query_norm;
        out[r] = norm > 0.0 ? static_cast<float>(dotp / norm) : 0.0f;
    });
}

// out[c] = (sum over r ascending of x[r, c]) / rows, in double, rounded once: the mean of each column of x [rows, cols].
inline void column_mean(const float* x, float* out, std::size_t rows, std::size_t cols) {
    parallel_for(cols, [&](std::size_t c) {
        double total = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            total += x[r * cols + c];
        }
        out[c] = rows > 0 ? static_cast<float>(total / static_cast<double>(rows)) : 0.0f;
    });
}

// Solves a x = b for a symmetric positive definite a [n, n] (only its lower triangle is read) and b [n, m], writing
// x [n, m]. Single-threaded, in double throughout: the Cholesky factor L (a = L L^T) column by column, j ascending,
// each entry's sum over k ascending; then forward substitution L y = b and back substitution L^T x = y, rows in
// order, each sum ascending. Results are rounded to float once at the end. Throws if a is not positive definite.
inline void cholesky_solve(const float* a, const float* b, float* x, std::size_t n, std::size_t m) {
    std::vector<double> l(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        double diagonal = a[j * n + j];
        for (std::size_t k = 0; k < j; ++k) {
            diagonal -= l[j * n + k] * l[j * n + k];
        }
        if (!(diagonal > 0.0)) {
            throw std::invalid_argument("matrix is not positive definite");
        }
        const double root = std::sqrt(diagonal);
        l[j * n + j] = root;
        for (std::size_t i = j + 1; i < n; ++i) {
            double value = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k) {
                value -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = value / root;
        }
    }
    std::vector<double> y(n * m, 0.0);
    for (std::size_t c = 0; c < m; ++c) {
        for (std::size_t i = 0; i < n; ++i) {
            double value = b[i * m + c];
            for (std::size_t k = 0; k < i; ++k) {
                value -= l[i * n + k] * y[k * m + c];
            }
            y[i * m + c] = value / l[i * n + i];
        }
        for (std::size_t ii = n; ii-- > 0;) {
            double value = y[ii * m + c];
            for (std::size_t k = ii + 1; k < n; ++k) {
                value -= l[k * n + ii] * y[k * m + c];
            }
            y[ii * m + c] = value / l[ii * n + ii];
        }
        for (std::size_t i = 0; i < n; ++i) {
            x[i * m + c] = static_cast<float>(y[i * m + c]);
        }
    }
}

}  // namespace dllm
