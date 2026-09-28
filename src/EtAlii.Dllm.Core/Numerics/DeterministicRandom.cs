namespace EtAlii.Dllm.Core.Numerics;

/// <summary>
/// A portable, seedable pseudo random number generator (xoshiro256**, seeded through SplitMix64).
/// Unlike <see cref="System.Random"/>, its output sequence is specified by this code alone and is
/// therefore identical on every runtime version, operating system and CPU architecture.
/// </summary>
public struct DeterministicRandom
{
    private ulong _s0, _s1, _s2, _s3;

    public DeterministicRandom(ulong seed)
    {
        var sm = seed;
        _s0 = SplitMix64(ref sm);
        _s1 = SplitMix64(ref sm);
        _s2 = SplitMix64(ref sm);
        _s3 = SplitMix64(ref sm);
    }

    /// <summary>Returns the next 64 random bits.</summary>
    public ulong NextUInt64()
    {
        var result = RotateLeft(_s1 * 5, 7) * 9;
        var t = _s1 << 17;
        _s2 ^= _s0;
        _s3 ^= _s1;
        _s1 ^= _s2;
        _s0 ^= _s3;
        _s2 ^= t;
        _s3 = RotateLeft(_s3, 45);
        return result;
    }

    /// <summary>Returns a double in [0, 1) built from the top 53 bits, so every value is exactly representable.</summary>
    public double NextDouble() => (NextUInt64() >> 11) * (1.0 / (1UL << 53));

    /// <summary>Returns a float in [0, 1) built from the top 24 bits.</summary>
    public float NextSingle() => (NextUInt64() >> 40) * (1.0f / (1U << 24));

    /// <summary>Returns an approximately normally distributed float (Irwin–Hall with 12 uniforms; no transcendental functions).</summary>
    public float NextGaussian()
    {
        var sum = 0.0;
        for (var i = 0; i < 12; i++)
        {
            sum += NextDouble();
        }
        return (float)(sum - 6.0);
    }

    private static ulong SplitMix64(ref ulong state)
    {
        var z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static ulong RotateLeft(ulong x, int k) => (x << k) | (x >> (64 - k));
}
