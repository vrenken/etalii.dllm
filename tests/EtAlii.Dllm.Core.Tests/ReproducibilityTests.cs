using EtAlii.Dllm.Core.Hosting;
using EtAlii.Dllm.Core.Sampling;

namespace EtAlii.Dllm.Core.Tests;

/// <summary>
/// These tests assert exact hashes. CI runs them on Linux, Windows and macOS (x64 and Arm64), so a pass everywhere
/// proves the pipeline is bit-exact across platforms, not just repeatable on one machine.
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
}
