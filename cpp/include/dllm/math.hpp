// Floating point kernels with a fixed evaluation order.
//
// Rules (see docs/research/deterministic-inference.md):
//  - Reductions run in one fixed, sequential order with a double accumulator; never an order chosen by
//    threads, batch size or the vector width of a BLAS library.
//  - exp is built from + - * / only, so a C library update cannot shift results.
//  - The extension is compiled with FMA contraction and fast-math disabled (see CMakeLists.txt).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace dllm {

inline double sum(const float* values, std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        acc += values[i];
    }
    return acc;
}

inline float dot(const float* a, const float* b, std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(acc);
}

// Index of the largest value; ties resolve to the lowest index.
inline std::size_t argmax(const float* values, std::size_t n) {
    if (n == 0) {
        throw std::invalid_argument("argmax of an empty array");
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

inline double power_of_two(int k) {
    const std::uint64_t bits = static_cast<std::uint64_t>(k + 1023) << 52;
    double result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// e^x: range reduction x = k*ln2 + r with a two-part ln2, a fixed degree-13 Taylor polynomial for e^r
// (Horner), then an exact scaling by 2^k. Accurate to a few ulps of double precision.
inline double exp(double x) {
    constexpr double ln2_hi = 6.93147180369123816490e-01;
    constexpr double ln2_lo = 1.90821492927058770002e-10;
    constexpr double inv_ln2 = 1.44269504088896338700e+00;

    if (x != x) {
        return x;
    }
    if (x > 709.78) {
        return std::numeric_limits<double>::infinity();
    }
    if (x < -745.2) {
        return 0.0;
    }

    const double kd = x * inv_ln2;
    const int k = static_cast<int>(kd >= 0 ? kd + 0.5 : kd - 0.5);
    const double r = (x - k * ln2_hi) - k * ln2_lo;

    double p = 1.0 / 6227020800.0;
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;

    // Split extreme exponents so every factor stays a normal double.
    int e = k;
    while (e > 1023) {
        p *= power_of_two(1023);
        e -= 1023;
    }
    while (e < -1022) {
        p *= power_of_two(-1022);
        e += 1022;
    }
    return p * power_of_two(e);
}

// Numerically stable softmax: subtract the maximum, exponentiate, normalise by the sequential sum.
inline void softmax(const float* logits, float* out, double* scratch, std::size_t n) {
    const double max = logits[argmax(logits, n)];
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        scratch[i] = dllm::exp(static_cast<double>(logits[i]) - max);
        total += scratch[i];
    }
    const double inv = 1.0 / total;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(scratch[i] * inv);
    }
}

}  // namespace dllm
