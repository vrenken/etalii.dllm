// Python bindings for the deterministic C++ kernels (module etalii_dllm._kernels).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <memory>
#include <vector>

#include "dllm/math.hpp"
#include "dllm/random.hpp"

namespace nb = nanobind;

using FloatVector = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OwnedFloatVector = nb::ndarray<nb::numpy, float, nb::ndim<1>>;

namespace {

OwnedFloatVector make_array(std::size_t n, float** data) {
    auto* buffer = new float[n];
    *data = buffer;
    nb::capsule owner(buffer, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return OwnedFloatVector(buffer, {n}, owner);
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
            auto result = make_array(n, &out);
            std::vector<double> scratch(n);
            dllm::softmax(logits.data(), out, scratch.data(), n);
            return result;
        },
        nb::arg("logits"), "Numerically stable softmax with a fixed evaluation order.");

    m.def(
        "fill_gaussian",
        [](std::uint64_t seed, std::size_t n) {
            float* out;
            auto result = make_array(n, &out);
            dllm::Random random(seed);
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = random.next_gaussian();
            }
            return result;
        },
        nb::arg("seed"), nb::arg("n"), "n approximately normal floats drawn from Random(seed).");
}
