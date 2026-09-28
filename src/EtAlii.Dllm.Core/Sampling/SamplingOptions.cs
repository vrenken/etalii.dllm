namespace EtAlii.Dllm.Core.Sampling;

/// <summary>
/// Controls token selection. With the same options, model and prompt the generated tokens are always identical:
/// a temperature above zero draws from a seeded <see cref="Numerics.DeterministicRandom"/>, never from ambient entropy.
/// </summary>
/// <param name="Temperature">0 means greedy decoding.</param>
/// <param name="TopK">Keep only the K most likely tokens; 0 disables.</param>
/// <param name="TopP">Nucleus sampling threshold in (0, 1]; 1 disables.</param>
/// <param name="Seed">Seed for the sampler's random stream.</param>
public sealed record SamplingOptions(float Temperature = 0f, int TopK = 0, float TopP = 1f, ulong Seed = 0)
{
    public static SamplingOptions Greedy { get; } = new();
}
