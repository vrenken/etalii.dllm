using EtAlii.Dllm.Core.Hosting;
using EtAlii.Dllm.Core.Sampling;

namespace EtAlii.Dllm.Core.Tests;

/// <summary>
/// These tests assert exact hashes, so any run-to-run drift fails the build. The current kernels are portable, so the
/// same values hold on every CI platform; that is a bonus, not a requirement (see docs/research/deterministic-inference.md).
/// </summary>
public class ReproducibilityTests
{
    [Fact]
    public void Weights_AreBitExact()
    {
        Assert.Equal(GoldenValues.SystemFingerprint, DllmEngine.CreateDefault().SystemFingerprint);
    }

    [Fact]
    public void GreedyGeneration_IsBitExact()
    {
        var result = DllmEngine.CreateDefault().Complete("Hello, world", 32, SamplingOptions.Greedy);
        Assert.Equal(GoldenValues.GreedyFingerprint, result.Fingerprint);
    }

    [Fact]
    public void SampledGeneration_IsBitExact()
    {
        var options = new SamplingOptions(Temperature: 0.8f, TopK: 40, TopP: 0.95f, Seed: 1234);
        var result = DllmEngine.CreateDefault().Complete("Hello, world", 32, options);
        Assert.Equal(GoldenValues.SampledFingerprint, result.Fingerprint);
    }

    [Fact]
    public void SameRequest_GivesSameOutput_AcrossEngineInstances()
    {
        var options = new SamplingOptions(Temperature: 1.0f, Seed: 99);
        var a = DllmEngine.CreateDefault().Complete("determinism", 64, options);
        var b = DllmEngine.CreateDefault().Complete("determinism", 64, options);
        Assert.Equal(a.Tokens, b.Tokens);
    }

    [Fact]
    public void DifferentSeeds_GiveDifferentOutput()
    {
        var engine = DllmEngine.CreateDefault();
        var a = engine.Complete("determinism", 64, new SamplingOptions(Temperature: 1.0f, Seed: 1));
        var b = engine.Complete("determinism", 64, new SamplingOptions(Temperature: 1.0f, Seed: 2));
        Assert.NotEqual(a.Tokens, b.Tokens);
    }

    [Fact]
    public async Task ConcurrentRequests_DoNotAffectEachOther()
    {
        var engine = DllmEngine.CreateDefault();
        var options = new SamplingOptions(Temperature: 0.9f, Seed: 7);
        var alone = engine.Complete("context window", 48, options);
        var concurrent = await Task.WhenAll(Enumerable.Range(0, 16).Select(i => Task.Run(() =>
            i % 2 == 0
                ? engine.Complete("context window", 48, options)
                : engine.Complete("other traffic " + i, 48, new SamplingOptions(Temperature: 1f, Seed: (ulong)i)))));
        for (var i = 0; i < concurrent.Length; i += 2)
        {
            Assert.Equal(alone.Tokens, concurrent[i].Tokens);
        }
    }
}
