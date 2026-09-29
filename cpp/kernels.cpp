// Python bindings for the deterministic C++ kernels (module etalii_dllm._kernels).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include <cmath>
#include <cstdint>
#include <new>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "dllm/cuda.hpp"
#include "dllm/fpenv.hpp"
#include "dllm/grad.hpp"
#include "dllm/math.hpp"
#include "dllm/nn.hpp"
#include "dllm/parallel.hpp"
#include "dllm/quant.hpp"
#include "dllm/simd.hpp"
#include "dllm/random.hpp"

namespace nb = nanobind;

using FloatVector = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using FloatTensor = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using DoubleVector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IndexVector = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OwnedFloatArray = nb::ndarray<nb::numpy, float>;
using MutableFloatTensor = nb::ndarray<float, nb::c_contig, nb::device::cpu>;
using Int8Matrix = nb::ndarray<const std::int8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using FloatMatrix = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using OwnedInt8Array = nb::ndarray<nb::numpy, std::int8_t>;

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

OwnedInt8Array make_int8_array(std::vector<std::size_t> shape, std::int8_t** data) {
    std::size_t n = 1;
    for (std::size_t d : shape) {
        n *= d;
    }
    void* buffer = ::operator new[](n == 0 ? kAlignment : n, std::align_val_t(kAlignment));
    *data = static_cast<std::int8_t*>(buffer);
    nb::capsule owner(buffer, [](void* p) noexcept { ::operator delete[](p, std::align_val_t(kAlignment)); });
    return OwnedInt8Array(buffer, shape.size(), shape.data(), owner);
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

// Every module function runs under FpEnvGuard (fpenv.hpp): the canonical floating point environment, whatever
// the caller's thread is set to.
struct GuardedModule {
    nb::module_& module;

    template <typename Func, typename... Extra>
    GuardedModule& def(const char* name, Func&& f, const Extra&... extra) {
        module.def(name, std::forward<Func>(f), extra..., nb::call_guard<dllm::FpEnvGuard>());
        return *this;
    }
    operator nb::module_&() { return module; }
    operator nb::handle() { return module; }
};

}  // namespace

NB_MODULE(_kernels, module) {
    GuardedModule m{module};
    module.doc() = "Deterministic numeric kernels for EtAlii.Dllm.";

    nb::class_<dllm::Random>(m, "Random", "xoshiro256** seeded through SplitMix64.")
        .def(nb::init<std::uint64_t>(), nb::arg("seed"))
        .def("next_u64", &dllm::Random::next_u64, nb::call_guard<dllm::FpEnvGuard>())
        .def("next_double", &dllm::Random::next_double, nb::call_guard<dllm::FpEnvGuard>())
        .def("next_gaussian", &dllm::Random::next_gaussian, nb::call_guard<dllm::FpEnvGuard>());

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
        "log_softmax",
        [](FloatVector logits) {
            const std::size_t n = logits.shape(0);
            float* out;
            auto result = make_array({n}, &out);
            dllm::log_softmax(logits.data(), out, n);
            return result;
        },
        nb::arg("logits"), "Log-softmax with a fixed evaluation order.");

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
            nb::gil_scoped_release release;
            dllm::linear(x.data(), w.data(), bias ? bias->data() : nullptr, out, leading_rows(x), in_features,
                         out_features);
            return result;
        },
        nb::arg("x"), nb::arg("weight"), nb::arg("bias").none() = nb::none(),
        "x[..., in] @ weight[out, in]^T (+ bias), fixed order, double accumulator.");

    m.def(
        "linear_reference",
        [](FloatTensor x, FloatTensor w, std::optional<FloatVector> bias) {
            require(x.ndim() >= 1 && w.ndim() == 2 && w.shape(1) == x.shape(x.ndim() - 1), "shapes do not match");
            require(!bias || bias->shape(0) == w.shape(0), "bias length must equal out_features");
            auto shape = shape_of(x);
            shape.back() = w.shape(0);
            float* out;
            auto result = make_array(shape, &out);
            dllm::linear_reference(x.data(), w.data(), bias ? bias->data() : nullptr, out, leading_rows(x),
                                   w.shape(1), w.shape(0));
            return result;
        },
        nb::arg("x"), nb::arg("weight"), nb::arg("bias").none() = nb::none(),
        "The single-threaded scalar statement of linear()'s order (for tests).");

    m.def(
        "pack_linear",
        [](FloatTensor w) {
            require(w.ndim() == 2, "weight must be [out_features, in_features]");
            const std::size_t out_features = w.shape(0);
            const std::size_t in_features = w.shape(1);
            float* out;
            auto result = make_array({dllm::panel_count(out_features), in_features, dllm::kPanel}, &out);
            nb::gil_scoped_release release;
            dllm::pack_linear(w.data(), out, in_features, out_features);
            return result;
        },
        nb::arg("weight"), "Weights [out, in] repacked into panels [ceil(out / 16), in, 16] for linear_packed.");

    m.def(
        "linear_packed",
        [](FloatTensor x, FloatTensor packed, std::size_t out_features, std::optional<FloatVector> bias) {
            require(x.ndim() >= 1, "x must have at least one dimension");
            const std::size_t in_features = x.shape(x.ndim() - 1);
            require(packed.ndim() == 3 && packed.shape(0) == dllm::panel_count(out_features) &&
                        packed.shape(1) == in_features && packed.shape(2) == dllm::kPanel,
                    "packed weight does not match x and out_features");
            require(!bias || bias->shape(0) == out_features, "bias length must equal out_features");
            auto shape = shape_of(x);
            shape.back() = out_features;
            float* out;
            auto result = make_array(shape, &out);
            nb::gil_scoped_release release;
            dllm::linear_packed(x.data(), packed.data(), bias ? bias->data() : nullptr, out, leading_rows(x),
                                in_features, out_features);
            return result;
        },
        nb::arg("x"), nb::arg("packed"), nb::arg("out_features"), nb::arg("bias").none() = nb::none(),
        "linear() with weights from pack_linear(); the same bits.");

    m.def(
        "quantize_q8_0",
        [](FloatMatrix w) {
            const std::size_t rows = w.shape(0);
            const std::size_t cols = w.shape(1);
            require(cols % dllm::kQ8Block == 0, "Q8_0 needs in_features to be a multiple of 32");
            std::int8_t* q;
            float* scales;
            auto q_array = make_int8_array({rows, cols}, &q);
            auto s_array = make_array({rows, cols / dllm::kQ8Block}, &scales);
            nb::gil_scoped_release release;
            for (std::size_t r = 0; r < rows; ++r) {
                dllm::quantize_q8_0(w.data() + r * cols, q + r * cols, scales + r * (cols / dllm::kQ8Block), cols);
            }
            return std::make_tuple(q_array, s_array);
        },
        nb::arg("weight"), "(int8 values [out, in], float32 scales [out, in / 32]) of a Q8_0 quantised matrix.");

    m.def(
        "linear_q8",
        [](FloatTensor x, Int8Matrix q, FloatMatrix scales, std::optional<FloatVector> bias) {
            require(x.ndim() >= 1, "x must have at least one dimension");
            const std::size_t in_features = x.shape(x.ndim() - 1);
            const std::size_t out_features = q.shape(0);
            require(q.shape(1) == in_features, "weight in_features does not match the last dimension of x");
            require(in_features % dllm::kQ8Block == 0, "Q8_0 needs in_features to be a multiple of 32");
            require(scales.shape(0) == out_features && scales.shape(1) == in_features / dllm::kQ8Block,
                    "scales must be [out_features, in_features / 32]");
            require(!bias || bias->shape(0) == out_features, "bias length must equal out_features");
            auto shape = shape_of(x);
            shape.back() = out_features;
            float* out;
            auto result = make_array(shape, &out);
            nb::gil_scoped_release release;
            dllm::linear_q8(x.data(), q.data(), scales.data(), bias ? bias->data() : nullptr, out, leading_rows(x),
                            in_features, out_features);
            return result;
        },
        nb::arg("x"), nb::arg("q"), nb::arg("scales"), nb::arg("bias").none() = nb::none(),
        "Q8_0 linear: activations quantised per 32-block, exact int32 block sums, combined in double in block order.");

    m.def(
        "set_threads", [](std::size_t n) { dllm::ThreadPool::global().set_threads(n); }, nb::arg("n"),
        "Number of kernel threads (0: DLLM_THREADS or the hardware concurrency). Never changes results.");
    m.def("threads", []() { return dllm::ThreadPool::global().threads(); }, "Current number of kernel threads.");
    module.def(  // unguarded: it reports the caller's state
        "fp_environment_is_canonical", []() { return dllm::fp_environment_is_canonical(); },
        "Whether the caller's floating point environment is the IEEE default (no flush-to-zero, round to nearest).");
    module.def(
        "_set_flush_to_zero",
        [](bool on) {
            dllm::set_flush_to_zero(on);
            return dllm::fp_environment_supported();
        },
        nb::arg("on"), "Turns flush-to-zero/denormals-are-zero on or off for the calling thread; for tests.");
    m.def("supported_isas", &dllm::supported_isas, "Instruction sets the dispatched kernels can use on this CPU.");
    m.def("isa", []() { return std::string(dllm::isa_name(dllm::active_isa())); },
          "Instruction set the dispatched kernels use.");
    m.def("set_isa", &dllm::set_isa, nb::arg("name"),
          "Forces an instruction set ('portable', 'avx2' or 'best'); for tests. Never changes results.");

    // CUDA backend (cuda.hpp): the same kernels on the GPU, with the same bits. etalii_dllm.cuda wraps these;
    // shapes are kept in Python, so the functions take sizes and check them against the arrays' byte counts.
    using dllm::cuda::Array;
    nb::class_<Array>(m, "CudaArray", "A range of GPU memory (a weight, an activation, a KV cache).")
        .def_prop_ro("bytes", &Array::bytes)
        .def("view", &Array::view, nb::arg("offset"), nb::arg("bytes"), "A byte range of the same memory.");

    m.def("cuda_device_count", &dllm::cuda::Runtime::device_count, "Number of CUDA devices the driver reports.");
    m.def(
        "cuda_initialize",
        [](const std::string& nvrtc_path, int device) {
            nb::gil_scoped_release release;
            dllm::cuda::Runtime::instance().initialize(nvrtc_path, device);
        },
        nb::arg("nvrtc_path"), nb::arg("device") = 0,
        "Loads the driver and NVRTC and compiles the kernels for `device` (only the first call does work).");
    m.def(
        "cuda_compile",
        [](const std::string& nvrtc_path, int arch) {
            dllm::cuda::CompiledKernels kernels;
            {
                nb::gil_scoped_release release;
                dllm::cuda::Compiler compiler;
                compiler.open(nvrtc_path);
                kernels = compiler.compile(arch);
            }
            return std::make_tuple(kernels.compiler, kernels.architecture,
                                   nb::bytes(kernels.image.data(), kernels.image.size()));
        },
        nb::arg("nvrtc_path"), nb::arg("arch"),
        "Compiles the GPU kernels with NVRTC for compute capability `arch` (e.g. 86) without a GPU; returns "
        "(compiler, architecture, image).");
    m.def(
        "cuda_info",
        []() -> std::optional<std::tuple<int, std::string, std::string, std::string>> {
            const auto& runtime = dllm::cuda::Runtime::instance();
            if (!runtime.ready()) {
                return std::nullopt;
            }
            return std::make_tuple(runtime.device_index(), runtime.device_name(), runtime.compiler(),
                                   runtime.architecture());
        },
        "(device index, device name, compiler, architecture) once initialised, else None.");

    m.def(
        "cuda_upload",
        [](nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> values) {
            const void* data = values.data();
            const std::size_t bytes = values.nbytes();
            nb::gil_scoped_release release;
            return dllm::cuda::upload(data, bytes);
        },
        nb::arg("values"), "Copies a C-contiguous array to the GPU.");
    m.def(
        "cuda_empty",
        [](std::size_t bytes) {
            nb::gil_scoped_release release;
            return dllm::cuda::empty(bytes);
        },
        nb::arg("bytes"), "Uninitialised GPU memory.");
    m.def(
        "cuda_download",
        [](const Array& array, std::vector<std::size_t> shape) {
            std::size_t n = 1;
            for (std::size_t d : shape) {
                n *= d;
            }
            require(n * sizeof(float) == array.bytes(), "shape does not match the array");
            float* out;
            auto result = make_array(shape, &out);
            nb::gil_scoped_release release;
            dllm::cuda::download(array, out);
            return result;
        },
        nb::arg("array"), nb::arg("shape"), "Copies a float32 array back from the GPU (waits for its producers).");
    m.def(
        "cuda_upload_linear",
        [](FloatMatrix w) {
            std::vector<float> wt = dllm::cuda::transpose_linear(w.data(), w.shape(1), w.shape(0));
            nb::gil_scoped_release release;
            return dllm::cuda::upload(wt.data(), wt.size() * sizeof(float));
        },
        nb::arg("weight"), "Uploads a linear weight [out, in] (transposed to [in, out]) for cuda_linear.");
    m.def(
        "cuda_upload_q8",
        [](Int8Matrix q, FloatMatrix scales) {
            const std::size_t out_features = q.shape(0);
            const std::size_t in_features = q.shape(1);
            require(in_features % dllm::kQ8Block == 0, "Q8_0 needs in_features to be a multiple of 32");
            require(scales.shape(0) == out_features && scales.shape(1) == in_features / dllm::kQ8Block,
                    "scales must be [out_features, in_features / 32]");
            std::vector<std::uint8_t> packed = dllm::cuda::pack_q8(q.data(), scales.data(), in_features, out_features);
            nb::gil_scoped_release release;
            return dllm::cuda::upload(packed.data(), packed.size());
        },
        nb::arg("q"), nb::arg("scales"), "Uploads Q8_0 weights (from quantize_q8_0) for cuda_linear_q8.");

    m.def(
        "cuda_linear",
        [](const Array& x, std::size_t rows, std::size_t in_features, const Array& weight, std::size_t out_features,
           std::optional<Array> bias) {
            nb::gil_scoped_release release;
            return dllm::cuda::linear(x, rows, in_features, weight, out_features, bias ? &*bias : nullptr);
        },
        nb::arg("x"), nb::arg("rows"), nb::arg("in_features"), nb::arg("weight"), nb::arg("out_features"),
        nb::arg("bias").none() = nb::none(), "linear() on the GPU; the same bits.");
    m.def(
        "cuda_linear_q8",
        [](const Array& x, std::size_t rows, std::size_t in_features, const Array& weight, std::size_t out_features,
           std::optional<Array> bias) {
            nb::gil_scoped_release release;
            return dllm::cuda::linear_q8(x, rows, in_features, weight, out_features, bias ? &*bias : nullptr);
        },
        nb::arg("x"), nb::arg("rows"), nb::arg("in_features"), nb::arg("weight"), nb::arg("out_features"),
        nb::arg("bias").none() = nb::none(), "linear_q8() on the GPU; the same bits.");
    m.def(
        "cuda_rms_norm",
        [](const Array& x, std::size_t rows, std::size_t dim, std::optional<Array> weight, double eps,
           bool add_unit_offset) {
            nb::gil_scoped_release release;
            return dllm::cuda::rms_norm(x, rows, dim, weight ? &*weight : nullptr, eps, add_unit_offset);
        },
        nb::arg("x"), nb::arg("rows"), nb::arg("dim"), nb::arg("weight").none(), nb::arg("eps"),
        nb::arg("add_unit_offset") = false, "rms_norm() on the GPU; the same bits.");
    m.def(
        "cuda_activation",
        [](const Array& x, int kind) {
            nb::gil_scoped_release release;
            return dllm::cuda::activation(x, kind);
        },
        nb::arg("x"), nb::arg("kind"), "silu (0), gelu (1) or gelu_tanh (2) on the GPU; the same bits.");
    m.def(
        "cuda_swiglu",
        [](const Array& gate, const Array& up) {
            nb::gil_scoped_release release;
            return dllm::cuda::swiglu(gate, up);
        },
        nb::arg("gate"), nb::arg("up"), "float32 silu(gate) * up on the GPU.");
    m.def(
        "cuda_add",
        [](const Array& a, const Array& b) {
            nb::gil_scoped_release release;
            return dllm::cuda::add(a, b);
        },
        nb::arg("a"), nb::arg("b"), "float32 a + b on the GPU.");
    m.def(
        "cuda_copy",
        [](const Array& source, const Array& destination) {
            nb::gil_scoped_release release;
            dllm::cuda::copy(source, destination);
        },
        nb::arg("source"), nb::arg("destination"), "Copies source into destination (same size) on the GPU.");
    m.def(
        "cuda_rope",
        [](const Array& x, const Array& positions, const Array& inv_freq, std::size_t tokens, std::size_t heads,
           std::size_t head_dim, bool interleaved, bool inverse) {
            nb::gil_scoped_release release;
            return dllm::cuda::rope(x, positions, inv_freq, tokens, heads, head_dim, interleaved, inverse);
        },
        nb::arg("x"), nb::arg("positions"), nb::arg("inv_freq"), nb::arg("tokens"), nb::arg("heads"),
        nb::arg("head_dim"), nb::arg("interleaved") = false, nb::arg("inverse") = false,
        "rope() on the GPU (positions int64, inv_freq float64); the same bits.");
    m.def(
        "cuda_attention",
        [](const Array& q, const Array& k, const Array& v, std::size_t q_len, std::size_t kv_len, std::size_t q_heads,
           std::size_t kv_heads, std::size_t head_dim, std::size_t value_dim, double scale, bool causal,
           std::size_t q_offset, std::size_t window) {
            nb::gil_scoped_release release;
            return dllm::cuda::attention(q, k, v, q_len, kv_len, q_heads, kv_heads, head_dim, value_dim, scale, causal,
                                         q_offset, window);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("q_len"), nb::arg("kv_len"), nb::arg("q_heads"),
        nb::arg("kv_heads"), nb::arg("head_dim"), nb::arg("value_dim"), nb::arg("scale"), nb::arg("causal"),
        nb::arg("q_offset"), nb::arg("window") = 0, "attention() on the GPU (reads the first kv_len keys and values); the same bits.");

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
        [](FloatTensor x, IndexVector positions, DoubleVector inv_freq, bool interleaved, bool inverse) {
            require(x.ndim() == 3, "x must be [tokens, heads, head_dim]");
            const std::size_t tokens = x.shape(0);
            const std::size_t head_dim = x.shape(2);
            const std::size_t rotary_dim = 2 * inv_freq.shape(0);
            require(positions.shape(0) == tokens, "positions must have one entry per token");
            require(rotary_dim <= head_dim, "2 * len(inv_freq) must not exceed head_dim");
            float* out;
            auto result = make_array(shape_of(x), &out);
            dllm::rope(x.data(), positions.data(), inv_freq.data(), out, tokens, x.shape(1), head_dim, rotary_dim,
                       interleaved, inverse);
            return result;
        },
        nb::arg("x"), nb::arg("positions"), nb::arg("inv_freq"), nb::arg("interleaved") = false,
        nb::arg("inverse") = false, "Rotary position embedding of x[tokens, heads, head_dim] (inverse: its transpose).");

    m.def(
        "attention",
        [](FloatTensor q, FloatTensor k, FloatTensor v, double scale, bool causal, std::int64_t q_offset,
           std::size_t window) {
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
            nb::gil_scoped_release release;
            dllm::attention(q.data(), k.data(), v.data(), out, q_len, kv_len, q.shape(1), k.shape(1), q.shape(2),
                            v.shape(2), scale, causal, static_cast<std::size_t>(q_offset), window);
            return result;
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("scale"), nb::arg("causal") = true,
        nb::arg("q_offset") = -1, nb::arg("window") = 0, "Scaled dot-product attention with grouped-query heads and a fixed order.");

    m.def(
        "linear_backward",
        [](FloatTensor x, FloatTensor w, FloatTensor dy, bool with_bias) {
            require(w.ndim() == 2, "weight must be [out_features, in_features]");
            const std::size_t in_features = w.shape(1);
            const std::size_t out_features = w.shape(0);
            require(x.ndim() >= 1 && x.shape(x.ndim() - 1) == in_features, "x does not match the weight");
            require(dy.ndim() == x.ndim() && dy.shape(dy.ndim() - 1) == out_features, "dy does not match the weight");
            const std::size_t rows = leading_rows(x);
            require(leading_rows(dy) == rows, "x and dy must have the same leading dimensions");
            float* dx;
            float* dw;
            float* db = nullptr;
            auto dx_array = make_array(shape_of(x), &dx);
            auto dw_array = make_array({out_features, in_features}, &dw);
            std::optional<OwnedFloatArray> db_array;
            if (with_bias) {
                db_array = make_array({out_features}, &db);
            }
            {
                nb::gil_scoped_release release;
                dllm::linear_backward(x.data(), w.data(), dy.data(), dx, dw, db, rows, in_features, out_features);
            }
            return std::make_tuple(dx_array, dw_array, db_array);
        },
        nb::arg("x"), nb::arg("weight"), nb::arg("dy"), nb::arg("with_bias") = false,
        "Gradients (dx, dweight, dbias or None) of linear(); fixed order, double accumulators.");

    m.def(
        "rms_norm_backward",
        [](FloatTensor x, std::optional<FloatVector> weight, FloatTensor dy, double eps) {
            require(x.ndim() >= 1, "x must have at least one dimension");
            const std::size_t dim = x.shape(x.ndim() - 1);
            require(dy.size() == x.size() && dy.shape(dy.ndim() - 1) == dim, "dy must have the shape of x");
            require(!weight || weight->shape(0) == dim, "weight length must equal the last dimension of x");
            float* dx;
            float* dw;
            auto dx_array = make_array(shape_of(x), &dx);
            auto dw_array = make_array({dim}, &dw);
            dllm::rms_norm_backward(x.data(), weight ? weight->data() : nullptr, dy.data(), dx, dw, leading_rows(x),
                                    dim, eps);
            return std::make_tuple(dx_array, dw_array);
        },
        nb::arg("x"), nb::arg("weight").none(), nb::arg("dy"), nb::arg("eps") = 1e-6,
        "Gradients (dx, dweight) of rms_norm(); fixed order, double accumulators.");

    m.def(
        "silu_backward",
        [](FloatTensor x, FloatTensor dy) {
            require(x.size() == dy.size(), "x and dy must have the same size");
            float* out;
            auto result = make_array(shape_of(x), &out);
            for (std::size_t i = 0; i < x.size(); ++i) {
                out[i] = dllm::silu_backward(x.data()[i], dy.data()[i]);
            }
            return result;
        },
        nb::arg("x"), nb::arg("dy"), "dy * silu'(x), elementwise.");

    m.def(
        "attention_backward",
        [](FloatTensor q, FloatTensor k, FloatTensor v, FloatTensor dout, double scale, bool causal,
           std::int64_t q_offset, std::size_t window) {
            require(q.ndim() == 3 && k.ndim() == 3 && v.ndim() == 3, "q, k and v must be [length, heads, dim]");
            const std::size_t q_len = q.shape(0);
            const std::size_t kv_len = k.shape(0);
            require(v.shape(0) == kv_len && v.shape(1) == k.shape(1), "k and v must have the same length and heads");
            require(q.shape(2) == k.shape(2), "q and k must have the same head_dim");
            require(k.shape(1) > 0 && q.shape(1) % k.shape(1) == 0, "q heads must be a multiple of kv heads");
            require(dout.ndim() == 3 && dout.shape(0) == q_len && dout.shape(1) == q.shape(1) &&
                        dout.shape(2) == v.shape(2),
                    "dout must be [q_len, q_heads, value_dim]");
            if (q_offset < 0) {
                q_offset = static_cast<std::int64_t>(kv_len) - static_cast<std::int64_t>(q_len);
            }
            require(q_offset >= 0, "q_offset must be non-negative");
            float* dq;
            float* dk;
            float* dv;
            auto dq_array = make_array(shape_of(q), &dq);
            auto dk_array = make_array(shape_of(k), &dk);
            auto dv_array = make_array(shape_of(v), &dv);
            dllm::attention_backward(q.data(), k.data(), v.data(), dout.data(), dq, dk, dv, q_len, kv_len,
                                     q.shape(1), k.shape(1), q.shape(2), v.shape(2), scale, causal,
                                     static_cast<std::size_t>(q_offset), window);
            return std::make_tuple(dq_array, dk_array, dv_array);
        },
        nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("dout"), nb::arg("scale"), nb::arg("causal") = true,
        nb::arg("q_offset") = -1, nb::arg("window") = 0, "Gradients (dq, dk, dv) of attention(); fixed order, double accumulators.");

    m.def(
        "cross_entropy",
        [](FloatTensor logits, IndexVector targets, double scale) {
            require(logits.ndim() == 2, "logits must be [rows, vocab]");
            require(targets.shape(0) == logits.shape(0), "targets must have one entry per row");
            float* dlogits;
            auto result = make_array(shape_of(logits), &dlogits);
            const double loss = dllm::cross_entropy(logits.data(), targets.data(), dlogits, logits.shape(0),
                                                    logits.shape(1), scale);
            return std::make_tuple(loss, result);
        },
        nb::arg("logits"), nb::arg("targets"), nb::arg("scale") = 1.0,
        "(summed loss, dlogits * scale) of softmax cross-entropy; negative targets are ignored.");

    m.def(
        "embedding_backward",
        [](FloatTensor dy, IndexVector tokens, std::size_t vocab) {
            require(dy.ndim() == 2 && dy.shape(0) == tokens.shape(0), "dy must be [len(tokens), dim]");
            float* out;
            auto result = make_array({vocab, dy.shape(1)}, &out);
            dllm::embedding_backward(dy.data(), tokens.data(), out, dy.shape(0), dy.shape(1), vocab);
            return result;
        },
        nb::arg("dy"), nb::arg("tokens"), nb::arg("vocabulary_size"),
        "Embedding gradient [vocab, dim]: rows of dy summed per token in position order.");

    m.def(
        "sum_squares", [](FloatTensor values) { return dllm::sum_squares(values.data(), values.size()); },
        nb::arg("values"), "Sum of squares in index order with a double accumulator.");

    m.def(
        "adamw_step",
        [](MutableFloatTensor param, FloatTensor grad, MutableFloatTensor m, MutableFloatTensor v, double lr,
           double beta1, double beta2, double eps, double weight_decay, double bias_correction1,
           double bias_correction2, double grad_scale) {
            const std::size_t n = param.size();
            require(grad.size() == n && m.size() == n && v.size() == n, "param, grad, m and v must have one size");
            dllm::adamw_step(param.data(), grad.data(), m.data(), v.data(), n, lr, beta1, beta2, eps, weight_decay,
                             bias_correction1, bias_correction2, grad_scale);
        },
        nb::arg("param"), nb::arg("grad"), nb::arg("m"), nb::arg("v"), nb::arg("lr"), nb::arg("beta1"),
        nb::arg("beta2"), nb::arg("eps"), nb::arg("weight_decay"), nb::arg("bias_correction1"),
        nb::arg("bias_correction2"), nb::arg("grad_scale") = 1.0, "One in-place AdamW step, element by element.");
}
