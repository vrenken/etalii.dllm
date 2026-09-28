namespace EtAlii.Dllm.Core.Tests;

/// <summary>
/// Reference outputs. RandomFirst/RandomSecond match the xoshiro256** reference implementation seeded with SplitMix64(42).
/// Change them only on purpose (new weights, new sampler semantics) and say so in the commit.
/// </summary>
internal static class GoldenValues
{
    public const ulong RandomFirst = 1546998764402558742UL;
    public const ulong RandomSecond = 6990951692964543102UL;
    public const string SystemFingerprint = "fp_aff631e39a75";
    public const string GreedyFingerprint = "706ce9aaacf0e839e9aa2f14318b1b7f72028e7cb0e23ecd7c8431008ff35e89";
    public const string SampledFingerprint = "cb6d23ee7267e79f4bcbb50f965626e05eb2f8219a8c26a6155f9e30f20b9cc6";
}
