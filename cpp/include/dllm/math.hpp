// Floating point kernels with a fixed evaluation order.
//
// Rules (see docs/research/deterministic-inference.md):
//  - Reductions run in one fixed, sequential order with a double accumulator; never an order chosen by
//    threads, batch size or the vector width of a BLAS library.
//  - exp is built from + - * / only, so a C library update cannot shift results.
//  - The extension is compiled with FMA contraction and fast-math disabled (see CMakeLists.txt).
//
// The CUDA backend compiles this file for the GPU as well (NVRTC with --fmad=false, see cuda.hpp): every function
// then runs the same IEEE operations in the same order on the device and returns the same bits. Under NVRTC the
// standard headers below are small shims supplied by cuda.hpp.
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
#ifndef __CUDACC_RTC__
    if (n == 0) {
        throw std::invalid_argument("argmax of an empty array");
    }
#endif
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

// Natural logarithm: x = m * 2^e with m in [sqrt(1/2), sqrt(2)), then log(m) = 2 atanh(s), s = (m-1)/(m+1), from a
// fixed 13-term odd series (|s| <= 0.172, so the last term is below 1e-19), plus e * ln2 with a two-part ln2.
inline double log(double x) {
    constexpr double ln2_hi = 6.93147180369123816490e-01;
    constexpr double ln2_lo = 1.90821492927058770002e-10;
    constexpr double sqrt_half = 7.07106781186547524401e-01;

    if (x != x || x < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (x == std::numeric_limits<double>::infinity()) {
        return x;
    }

    int e = 0;
    // Bring subnormals into the normal range before reading the exponent bits.
    if (x < std::numeric_limits<double>::min()) {
        x *= power_of_two(54);
        e -= 54;
    }
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    e += static_cast<int>((bits >> 52) & 0x7ff) - 1023;
    bits = (bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m;
    std::memcpy(&m, &bits, sizeof(m));
    // m is in [1, 2); fold it into [sqrt(1/2), sqrt(2)) so |s| stays small.
    if (m >= 2.0 * sqrt_half) {
        m *= 0.5;
        e += 1;
    }

    const double s = (m - 1.0) / (m + 1.0);
    const double s2 = s * s;
    double p = 1.0 / 25.0;
    p = p * s2 + 1.0 / 23.0;
    p = p * s2 + 1.0 / 21.0;
    p = p * s2 + 1.0 / 19.0;
    p = p * s2 + 1.0 / 17.0;
    p = p * s2 + 1.0 / 15.0;
    p = p * s2 + 1.0 / 13.0;
    p = p * s2 + 1.0 / 11.0;
    p = p * s2 + 1.0 / 9.0;
    p = p * s2 + 1.0 / 7.0;
    p = p * s2 + 1.0 / 5.0;
    p = p * s2 + 1.0 / 3.0;
    const double log_m = 2.0 * s + 2.0 * s * s2 * p;
    return (e * ln2_hi + log_m) + e * ln2_lo;
}

namespace detail {

// sin and cos of r in [-pi/4, pi/4] from fixed Taylor polynomials (Horner in r^2); the first omitted terms are
// below 1e-19.
inline double sin_kernel(double r) {
    const double r2 = r * r;
    double p = -1.0 / 1307674368000.0;           // -1/15!
    p = p * r2 + 1.0 / 6227020800.0;             // 1/13!
    p = p * r2 - 1.0 / 39916800.0;               // -1/11!
    p = p * r2 + 1.0 / 362880.0;                 // 1/9!
    p = p * r2 - 1.0 / 5040.0;                   // -1/7!
    p = p * r2 + 1.0 / 120.0;                    // 1/5!
    p = p * r2 - 1.0 / 6.0;                      // -1/3!
    return r + r * r2 * p;
}

inline double cos_kernel(double r) {
    const double r2 = r * r;
    double p = 1.0 / 20922789888000.0;           // 1/16!
    p = p * r2 - 1.0 / 87178291200.0;            // -1/14!
    p = p * r2 + 1.0 / 479001600.0;              // 1/12!
    p = p * r2 - 1.0 / 3628800.0;                // -1/10!
    p = p * r2 + 1.0 / 40320.0;                  // 1/8!
    p = p * r2 - 1.0 / 720.0;                    // -1/6!
    p = p * r2 + 1.0 / 24.0;                     // 1/4!
    return (1.0 - 0.5 * r2) + r2 * r2 * p;
}

// x = n * pi/2 + r with |r| <= pi/4 (Cody-Waite, the three 33-bit parts of pi/2 from fdlibm, so n * part is exact
// for |n| < 2^20, i.e. |x| below about 1.6e6). Larger arguments lose accuracy but stay deterministic everywhere.
inline double reduce_half_pi(double x, int* quadrant) {
    constexpr double two_over_pi = 6.36619772367581382433e-01;
    constexpr double pio2_1 = 1.57079632673412561417e+00;
    constexpr double pio2_2 = 6.07710050630396597660e-11;
    constexpr double pio2_3 = 2.02226624871116645580e-21;
    constexpr double pio2_3t = 8.47842766036889956997e-32;

    constexpr double two_52 = 4503599627370496.0;
    constexpr double two_62 = 4611686018427387904.0;

    const double nd = x * two_over_pi;
    // From 2^52 on nd is already an integer; it is used as is, because converting a double beyond 2^63 to an integer
    // gives different results on x86-64 and arm64.
    double n = nd;
    if (nd < two_52 && nd > -two_52) {
        n = nd >= 0 ? static_cast<double>(static_cast<long long>(nd + 0.5))
                    : static_cast<double>(static_cast<long long>(nd - 0.5));
    }
    const double r = (((x - n * pio2_1) - n * pio2_2) - n * pio2_3) - n * pio2_3t;
    // n mod 4; from 2^62 on n is a multiple of 4.
    *quadrant = (n < two_62 && n > -two_62) ? static_cast<int>(static_cast<long long>(n) & 3) : 0;
    return r;
}

}  // namespace detail

inline double sin(double x) {
    if (x != x || x == std::numeric_limits<double>::infinity() || x == -std::numeric_limits<double>::infinity()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    int quadrant;
    const double r = detail::reduce_half_pi(x, &quadrant);
    switch (quadrant) {
        case 0: return detail::sin_kernel(r);
        case 1: return detail::cos_kernel(r);
        case 2: return -detail::sin_kernel(r);
        default: return -detail::cos_kernel(r);
    }
}

inline double cos(double x) {
    if (x != x || x == std::numeric_limits<double>::infinity() || x == -std::numeric_limits<double>::infinity()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    int quadrant;
    const double r = detail::reduce_half_pi(x, &quadrant);
    switch (quadrant) {
        case 0: return detail::cos_kernel(r);
        case 1: return -detail::sin_kernel(r);
        case 2: return -detail::cos_kernel(r);
        default: return detail::sin_kernel(r);
    }
}

// atan from a fixed Taylor polynomial: |x| > 1 uses atan(x) = pi/2 - atan(1/x), and x above tan(pi/12) uses
// atan(x) = pi/6 + atan((sqrt(3) x - 1) / (sqrt(3) + x)), so the series only sees |r| <= 2 - sqrt(3) (about 0.268),
// where the first omitted term is below 1e-19.
inline double atan(double x) {
    if (x != x) {
        return x;
    }
    if (x < 0) {
        return -atan(-x);
    }
    constexpr double half_pi = 1.57079632679489661923;
    constexpr double sixth_pi = 0.52359877559829887308;
    constexpr double sqrt3 = 1.73205080756887729353;
    constexpr double tan_twelfth_pi = 0.26794919243112270647;
    if (x > 1.0) {
        return half_pi - atan(1.0 / x);
    }
    double offset = 0.0;
    double r = x;
    if (x > tan_twelfth_pi) {
        offset = sixth_pi;
        r = (sqrt3 * x - 1.0) / (sqrt3 + x);
    }
    const double r2 = r * r;
    double p = -1.0 / 35.0;
    for (int n = 16; n >= 1; --n) {  // Horner in r^2: sum of (-1)^n r^(2n+1) / (2n+1), n = 1..17
        p = p * r2 + ((n % 2) ? -1.0 : 1.0) / (2.0 * n + 1.0);
    }
    return offset + (r + r * r2 * p);
}

// acos(x) = 2 atan(sqrt((1 - x) / (1 + x))) on [-1, 1]; NaN outside.
inline double acos(double x) {
    if (!(x >= -1.0 && x <= 1.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x == -1.0) {
        return 3.14159265358979323846;
    }
#ifdef __CUDACC_RTC__
    const double root = __dsqrt_rn((1.0 - x) / (1.0 + x));  // NVRTC has no <cmath>; both are the IEEE square root
#else
    const double root = std::sqrt((1.0 - x) / (1.0 + x));
#endif
    return 2.0 * atan(root);
}

// tanh from exp: tanh(x) = sign(x) * (1 - 2 / (e^(2|x|) + 1)); a Taylor polynomial near 0 avoids cancellation.
inline double tanh(double x) {
    if (x != x) {
        return x;
    }
    const double a = x < 0 ? -x : x;
    double t;
    if (a < 0.125) {
        // tanh(a) = a - a^3/3 + 2a^5/15 - 17a^7/315 + 62a^9/2835 - 1382a^11/155925 + 21844a^13/6081075 - ...
        const double a2 = a * a;
        double p = -929569.0 / 638512875.0;
        p = p * a2 + 21844.0 / 6081075.0;
        p = p * a2 - 1382.0 / 155925.0;
        p = p * a2 + 62.0 / 2835.0;
        p = p * a2 - 17.0 / 315.0;
        p = p * a2 + 2.0 / 15.0;
        p = p * a2 - 1.0 / 3.0;
        t = a + a * a2 * p;
    } else if (a > 22.0) {
        t = 1.0;
    } else {
        t = 1.0 - 2.0 / (dllm::exp(2.0 * a) + 1.0);
    }
    return x < 0 ? -t : t;
}

// Logistic sigmoid 1 / (1 + e^-x), evaluated so the exponent is never large and positive.
inline double sigmoid(double x) {
    if (x >= 0) {
        return 1.0 / (1.0 + dllm::exp(-x));
    }
    const double e = dllm::exp(x);
    return e / (1.0 + e);
}

namespace detail {

// erfc(a) for a >= 2.5: e^(-a^2) / sqrt(pi) * K(a) with K the Laplace continued fraction
// 1 / (a + (1/2) / (a + 1 / (a + (3/2) / (a + ...)))), evaluated backwards from a fixed depth of 80.
inline double erfc_tail(double a) {
    constexpr double inv_sqrt_pi = 5.64189583547756286948e-01;
    double k = a;
    for (int n = 80; n >= 1; --n) {
        k = a + (0.5 * n) / k;
    }
    return dllm::exp(-a * a) * inv_sqrt_pi / k;
}

}  // namespace detail

// Error function. |x| < 2.5: the Maclaurin series erf(x) = 2/sqrt(pi) sum (-1)^n x^(2n+1) / (n! (2n+1)) with a fixed
// 60 terms; otherwise 1 - erfc(|x|) from the continued fraction (exactly 1 beyond |x| = 6).
inline double erf(double x) {
    constexpr double two_over_sqrt_pi = 1.12837916709551257390e+00;

    if (x != x) {
        return x;
    }
    const double a = x < 0 ? -x : x;
    double result;
    if (a < 2.5) {
        const double a2 = a * a;
        double term = a;  // (-1)^n a^(2n+1) / n!
        double total = a;
        for (int n = 1; n < 60; ++n) {
            term = -term * a2 / n;
            total += term / (2 * n + 1);
        }
        result = two_over_sqrt_pi * total;
    } else if (a > 6.0) {
        result = 1.0;
    } else {
        result = 1.0 - detail::erfc_tail(a);
    }
    return x < 0 ? -result : result;
}

// Complementary error function 1 - erf(x), accurate in relative terms for large positive x (no cancellation).
inline double erfc(double x) {
    if (x != x) {
        return x;
    }
    if (x < 2.5) {
        return 1.0 - dllm::erf(x);
    }
    if (x > 27.3) {
        return 0.0;
    }
    return detail::erfc_tail(x);
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

// Log-softmax: (l_i - max) - log(sum_j exp(l_j - max)), the sum over j ascending in double; each result is rounded
// once to float. Gives the log-probabilities reported by the API (logprobs).
inline void log_softmax(const float* logits, float* out, std::size_t n) {
    const double max = logits[argmax(logits, n)];
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        total += dllm::exp(static_cast<double>(logits[i]) - max);
    }
    const double log_total = dllm::log(total);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>((static_cast<double>(logits[i]) - max) - log_total);
    }
}

}  // namespace dllm
