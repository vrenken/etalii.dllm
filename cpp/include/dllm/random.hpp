// Portable, seedable pseudo random number generator: xoshiro256** seeded through SplitMix64.
// Its output sequence is fixed by this code alone, unlike std::mt19937 distributions or numpy's
// default generator, whose derived values may change between library versions.
#pragma once

#include <cstdint>

namespace dllm {

class Random {
public:
    explicit Random(std::uint64_t seed) {
        std::uint64_t sm = seed;
        s0_ = split_mix64(sm);
        s1_ = split_mix64(sm);
        s2_ = split_mix64(sm);
        s3_ = split_mix64(sm);
    }

    std::uint64_t next_u64() {
        const std::uint64_t result = rotl(s1_ * 5, 7) * 9;
        const std::uint64_t t = s1_ << 17;
        s2_ ^= s0_;
        s3_ ^= s1_;
        s1_ ^= s2_;
        s0_ ^= s3_;
        s2_ ^= t;
        s3_ = rotl(s3_, 45);
        return result;
    }

    // Double in [0, 1) from the top 53 bits, so every value is exactly representable.
    double next_double() { return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0); }

    // Approximately normal float (Irwin-Hall with 12 uniforms; no transcendental functions).
    float next_gaussian() {
        double sum = 0.0;
        for (int i = 0; i < 12; ++i) {
            sum += next_double();
        }
        return static_cast<float>(sum - 6.0);
    }

private:
    static std::uint64_t split_mix64(std::uint64_t& state) {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    std::uint64_t s0_, s1_, s2_, s3_;
};

}  // namespace dllm
