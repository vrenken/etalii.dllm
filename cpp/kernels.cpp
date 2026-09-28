// Python bindings for the deterministic C++ kernels (module etalii_dllm._kernels).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <cmath>
#include <cstdint>
#include <new>
#include <optional>
#include <vector>

#include "dllm/math.hpp"
#include "dllm/nn.hpp"
#include "dllm/random.hpp"

namespace nb = nanobind;

using FloatVector = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using FloatTensor = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using DoubleVector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IndexVector = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OwnedFloatArray = nb::ndarray<nb::numpy, float>;

namespace {

// Kernel outputs are 64-byte aligned (a cache line, and the widest SIMD register), like etalii_dllm.tensor.Tensor.
constexpr std::size_t kAlignment = 64;

OwnedFloatArray make_array(std::vector<std::size_t> shape, float** data) {
    std::size_t n = 1;
    for (std::size_t d : shape) {
        n *= d;
    }
    void* buffer = ::operator new[](n == 0 ? kAlignment : n * sizeof(float), std::align_val_t(kAlignment));
    *data = static_cast<float*>(buffer);
    nb::capsule owner(buffer, [](void* p) noexcept { ::operator delete[](p, std::align_val_t(kAlignment)); });
    return OwnedFloatArray(buffer, shape.size(), shape.data(), owner);
}

std::vector<std::size_t> shape_of(const FloatTensor& t) {
    std::vector<std::size_t> shape(t.ndim());
    for (std::size_t i = 0; i < t.ndim(); ++i) {
        shape[i] = t.shape(i);
    }
    return shape;
}

// Number of rows when the last dimension is the feature dimension.
std::size_t leading_rows(const FloatTensor& t) {
    std::size_t rows = 1;
    for (std::size_t i = 0; i + 1 < t.ndim(); ++i) {
        rows *= t.shape(i);
    }
    return rows;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw nb::value_error(message);
    }
}

template <float (*F)(float)>
OwnedFloatArray elementwise(FloatTensor x) {
    float* out;
    auto result = make_array(shape_of(x), &out);
    const float* in = x.data();
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = F(in[i]);
    }
    return result;
}

}  // namespace

NB_MODULE(_kernels, m) {
    m.doc() = "Deterministic numeric kernels for EtAlii.Dllm.";

    nb::class_<dllm::Random>(m, "Random", "xoshiro256** seeded through SplitMix64.")
        .def(nb::init<std::uint64_t>(), nb::arg("seed"))
        .def("next_u64", &dllm::Random::next_u64)
        .def("next_double", &dllm::Random::next_double)
        .def("next_gaussian", &dllm::Random::next_gaussian);

    m.def("exp", &dllm::exp, nb::arg("x"), "Portable e^x built from basic IEEE operations.");
    m.def("log", &dllm::log, nb::arg("x"), "Portable natural logarithm built from basic IEEE operations.");
    m.def("sin", &dllm::sin, nb::arg("x"), "Portable sine built from basic IEEE operations.");
    m.def("cos", &dllm::cos, nb::arg("x"), "Portable cosine built from basic IEEE operations.");
    m.def("tanh", &dllm::tanh, nb::arg("x"), "Portable hyperbolic tangent built on dllm exp.");
    m.def("sigmoid", &dllm::sigmoid, nb::arg("x"), "Portable logistic sigmoid built on dllm exp.");
    m.def("erf", &dllm::erf, nb::arg("x"), "Portable error function built from basic IEEE operations.");
    m.def("erfc", &dllm::erfc, nb::arg("x"), "Portable complementary error function 1 - erf(x).");

    m.def("sum", [](FloatVector v) { return dllm::sum(v.data(), v.shape(0)); }, nb::arg("values"),
          "Sum in index order with a double accumulator.");

    m.def(
        "dot",
        [](FloatVector a, FloatVector b) {
            if (a.shape(0) != b.shape(0)) {
                throw nb::value_error("vectors must have the same length");
            }
            return dllm::dot(a.data(), b.data(), a.shape(0));
        },
        nb::arg("a"), nb::arg("b"), "Dot product in index order with a double accumulator.");

    m.def("argmax", [](FloatVector v) { return dllm::argmax(v.data(), v.shape(0)); }, nb::arg("values"),
          "Index of the largest value; ties resolve to the lowest index.");

    m.def(
        "softmax",
        [](FloatVector logits) {
            const std::size_t n = logits.shape(0);
            float* out;
            auto result = make_array({n}, &out);
            std::vector<double> scratch(n);
            dllm::softmax(logits.data(), out, scratch.data(), n);
            return result;
        },
        nb::arg("logits"), "Numerically stable softmax with a fixed evaluation order.");

    m.def(
        "fill_gaussian",
        [](std::uint64_t seed, std::size_t n) {
            float* out;
            auto result = make_array({n}, &out);
            dllm::Random random(seed);
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = random.next_gaussian();
            }
            return result;
        },
        nb::arg("seed"), nb::arg("n"), "n approximately normal floats drawn from Random(seed).");

    m.def(
        "linear",
        [](FloatTensor x, FloatTensor w, std::optional<FloatVector> bias) {
            require(x.ndim() >= 1, "x must have at least one dimension");
            require(w.ndim() == 2, "weight must be [out_features, in_features]");
            const std::size_t in_features = x.shape(x.ndim() - 1);
            const std::size_t out_features = w.shape(0);
            require(w.shape(1) == in_features, "weight in_features does not match the last dimension of x");
            require(!bias || bias->shape(0) == out_features, "bias length must equal out_features");
            auto shape = shape_of(x);
            shape.back() = out_features;
            float* out;
            auto result = make_array(shape, &out);
            dllm::linear(x.data(), w.data(), bias ? bias->data() : nullptr, out, leading_rows(x), in_features,
                         out_features);
            return result;
        },
        nb::arg("x"), nb::arg("weight"), nb::arg("bias").none() = nb::none(),
        "x[..., in] @ weight[out, in]^T (+ bias), fixed order, double accumulator.");

    m.def(
        "matmul",
        [](FloatTensor a, FloatTensor b) {
            require(a.ndim() == 2 && b.ndim() == 2, "matmul takes two matrices");
            require(a.shape(1) == b.shape(0), "inner dimensions must match");
            float* out;
            auto result = make_array({a.shape(0), b.shape(1)}, &out);
            dllm::matmul(a.data(), b.data(), out, a.shape(0), a.shape(1), b.shape(1));
            return result;
        },
        nb::arg("a"), nb::arg("b"), "a[m, k] @ b[k, n], fixed order, double accumulator.");

    m.def(
        "rms_norm",
        [](FloatTensor x, std::optional<FloatVector> weight, double eps, bool add_unit_offset) {
            require(x.ndim() >= 1, "x must have at least one dimension");
            const std::size_t dim = x.shape(x.ndim() - 1);
            require(!weight || weight->shape(0) == dim, "weight length must equal the last dimension of x");
            float* out;
            auto result = make_array(shape_of(x), &out);
            dllm::rms_norm(x.data(), weight ? weight->data() : nullptr, out, leading_rows(x), dim, eps,
                           add_unit_offset);
            return result;
        },
        nb::arg("x"), nb::arg("weight").none() = nb::none(), nb::arg("eps") = 1e-6,
        nb::arg("add_unit_offset") = false, "RMSNorm over the last dimension, fixed order, double accumulator.");

    m.def("silu", &elementwise<dllm::silu>, nb::arg("x"), "Elementwise x * sigmoid(x).");
    m.def("gelu", &elementwise<dllm::gelu>, nb::arg("x"), "Elementwise exact (erf) GELU.");
    m.def("gelu_tanh", &elementwise<dllm::gelu_tanh>, nb::arg("x"), "Elementwise tanh-approximated GELU.");

    m.def(
        "rope",
        [](FloatTensor x, IndexVector positions, DoubleVector inv_freq, bool interleaved) {
            require(x.ndim() == 3, "x must be [tokens, heads, head_dim]");
            const std::size_t tokens = x.shape(0);
            const std::size_t head_dim = x.shape(2);
            const std::size_t rotary_dim = 2 * inv_freq.shape(0);
            require(positions.shape(0) == tokens, "positions must have one entry per token");
            require(rotary_dim <= head_dim, "2 * len(inv_freq) must not exceed head_dim");
            float* out;
            auto result = make_array(shape_of(x), &out);
            dllm::rope(x.data(), positions.data(), inv_freq.data(), out, tokens, x.shape(1), head_dim, rotary_dim,
                       interleaved);
            return result;
        },
        nb::arg("x"), nb::arg("positions"), nb::arg("inv_freq"), nb::arg("interleaved") = false,
        "Rotary position embedding of x[tokens, heads, head_dim].");

    m.def(
        "attention",
        [](FloatTensor q, FloatTensor k, FloatTensor v, double scale, bool causal, std::int64_t q_offset) {
            require(q.ndim() == 3 && k.ndim() == 3 && v.ndim() == 3, "q, k and v must be [length, heads, dim]");
            const std::size_t q_len = q.shape(0);
            const std::size_t kv_len = k.shape(0);
            require(v.shape(0) == kv_len && v.shape(1) == k.shape(1), "k and v must have the same length and heads");
            require(q.shape(2) == k.shape(2), "q and k must have the same head_dim");
            require(k.shape(1) > 0 && q.shape(1) % k.shape(1) == 0, "q heads must be a multiple of kv heads");
            if (q_offset < 0) {
                q_offset = static_cast<std::int64_t>(kv_len) - static_cast<std::int64_t>(q_len);
            }
            require(q_offset >= 0, "q_offset must be non-negative");
            float* out;
            auto result = make_array({q_len, q.shape(1), v.shape(2)}, &out);
            dllm::attention(q.data(), k.data(), v.data(), out, q_len, kv_len, q.shape(1), k.shape(1), q.shape(2),
                            v.shape(2), scale, causal, static_cast<std::size_t>(q_offset));
            return result;
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("scale"), nb::arg("causal") = true,
        nb::arg("q_offset") = -1, "Scaled dot-product attention with grouped-query heads and a fixed order.");
}
