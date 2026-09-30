// The floating point environment every kernel runs in.
//
// IEEE 754 fixes the result of + - * / and sqrt only for a given rounding mode and subnormal handling. The kernels
// assume the default: round to nearest even, subnormals kept (no flush-to-zero, no denormals-are-zero), all
// exceptions masked, NaN payloads propagated. The process can be in another state without our knowing: a library
// built with -ffast-math turns flush-to-zero on when it loads (crtfastmath.o), and a host application may change the
// rounding mode. FpEnvGuard puts the calling thread into the canonical state for the duration of a kernel call and
// restores the caller's state afterwards; the thread pool's workers set it once when they start. Cost: a control
// register read (and, only when it differs, two writes) per kernel call. See docs/kernels.md ("Floating point
// environment").
#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#define DLLM_FPENV_X86 1
#include <xmmintrin.h>
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#define DLLM_FPENV_ARM64 1
#endif

namespace dllm {

#if defined(DLLM_FPENV_X86)
using FpControl = unsigned int;

// MXCSR: bits 0-5 are sticky status flags (kept), 7-12 the exception masks (all set), 13-14 the rounding mode
// (0: nearest), 6 DAZ and 15 FTZ (both clear).
inline FpControl canonical_fp_control(FpControl current) { return (current & 0x3Fu) | 0x1F80u; }
inline FpControl read_fp_control() { return _mm_getcsr(); }
inline void write_fp_control(FpControl value) { _mm_setcsr(value); }
inline bool fp_environment_supported() { return true; }
inline void set_flush_to_zero(bool on) {
    const FpControl csr = read_fp_control();
    write_fp_control(on ? (csr | 0x8040u) : (csr & ~0x8040u));
}
#elif defined(DLLM_FPENV_ARM64)
using FpControl = std::uint64_t;

// FPCR: bits 8-15 trap enables, 19 FZ16, 22-23 rounding mode (0: nearest), 24 FZ, 25 DN (default NaN), 26 AHP; all
// clear. Other bits are kept.
constexpr FpControl kArmFpcrMask = (0xFFull << 8) | (1ull << 19) | (0x1Full << 22);
inline FpControl canonical_fp_control(FpControl current) { return current & ~kArmFpcrMask; }
inline FpControl read_fp_control() {
    FpControl value;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(value));
    return value;
}
inline void write_fp_control(FpControl value) { __asm__ __volatile__("msr fpcr, %0" : : "r"(value)); }
inline bool fp_environment_supported() { return true; }
inline void set_flush_to_zero(bool on) {
    const FpControl fpcr = read_fp_control();
    write_fp_control(on ? (fpcr | (1ull << 24)) : (fpcr & ~(1ull << 24)));
}
#else
// Other targets: the environment is left alone (and reported as canonical).
using FpControl = unsigned int;
inline FpControl canonical_fp_control(FpControl current) { return current; }
inline FpControl read_fp_control() { return 0; }
inline void write_fp_control(FpControl) {}
inline bool fp_environment_supported() { return false; }
inline void set_flush_to_zero(bool) {}
#endif

// Whether the calling thread is in the canonical state (the caller's state, outside a kernel call).
inline bool fp_environment_is_canonical() {
    const FpControl current = read_fp_control();
    return canonical_fp_control(current) == current;
}

// Puts the calling thread into the canonical state; permanent (for threads the kernels own).
inline void enter_canonical_fp_environment() {
    const FpControl current = read_fp_control();
    const FpControl canonical = canonical_fp_control(current);
    if (canonical != current) {
        write_fp_control(canonical);
    }
}

// Canonical state for one scope, then the caller's state again (sticky status flags raised inside are kept).
class FpEnvGuard {
public:
    FpEnvGuard() : saved_(read_fp_control()) {
        const FpControl canonical = canonical_fp_control(saved_);
        changed_ = canonical != saved_;
        if (changed_) {
            write_fp_control(canonical);
        }
    }
    ~FpEnvGuard() {
        if (changed_) {
#if defined(DLLM_FPENV_X86)
            write_fp_control((saved_ & ~0x3Fu) | (read_fp_control() & 0x3Fu));
#else
            write_fp_control(saved_);
#endif
        }
    }
    FpEnvGuard(const FpEnvGuard&) = delete;
    FpEnvGuard& operator=(const FpEnvGuard&) = delete;

private:
    FpControl saved_;
    bool changed_ = false;
};

}  // namespace dllm
