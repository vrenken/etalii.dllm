// Instruction-set dispatch for the hot kernels.
//
// Hot loops vectorise across *independent* accumulators (one per output element), never inside one accumulation.
// Each has a baseline variant (SSE2 on x86-64, NEON on arm64, plain C++ elsewhere) and, on x86-64 with GCC or
// Clang, an AVX2 + FMA variant compiled through a function target attribute. All variants perform the same IEEE
// operations in the same order per output, so they give identical bits; the widest one the CPU supports is picked
// once at start-up (the code path is fixed per machine) and tests force each one in turn (set_isa). The floating
// point flags in CMakeLists.txt forbid the compiler from contracting anything into FMA on its own.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define DLLM_X86_DISPATCH 1
#include <immintrin.h>
#define DLLM_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif

// The baseline vector unit, available on every CPU of the architecture without dispatch.
#if defined(__x86_64__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#define DLLM_SSE2 1
#include <emmintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define DLLM_NEON 1
#include <arm_neon.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DLLM_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define DLLM_ALWAYS_INLINE __forceinline
#else
#define DLLM_ALWAYS_INLINE inline
#endif

namespace dllm {

enum class Isa : int { portable = 0, avx2 = 1 };

inline const char* isa_name(Isa isa) {
    switch (isa) {
        case Isa::avx2:
            return "avx2";
        default:
            return "portable";
    }
}

inline bool isa_supported(Isa isa) {
    if (isa == Isa::portable) {
        return true;
    }
#ifdef DLLM_X86_DISPATCH
    __builtin_cpu_init();
    if (isa == Isa::avx2) {
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }
#endif
    return false;
}

inline std::vector<std::string> supported_isas() {
    std::vector<std::string> names;
    for (Isa isa : {Isa::portable, Isa::avx2}) {
        if (isa_supported(isa)) {
            names.emplace_back(isa_name(isa));
        }
    }
    return names;
}

inline Isa best_isa() {
    return isa_supported(Isa::avx2) ? Isa::avx2 : Isa::portable;
}

// The instruction set the dispatched kernels use; fixed at start-up unless a test overrides it.
inline Isa& active_isa() {
    static Isa isa = best_isa();
    return isa;
}

inline void set_isa(const std::string& name) {
    for (Isa isa : {Isa::portable, Isa::avx2}) {
        if (name == isa_name(isa)) {
            if (!isa_supported(isa)) {
                throw std::invalid_argument("instruction set not supported on this CPU: " + name);
            }
            active_isa() = isa;
            return;
        }
    }
    if (name == "best") {
        active_isa() = best_isa();
        return;
    }
    throw std::invalid_argument("unknown instruction set: " + name);
}

}  // namespace dllm
