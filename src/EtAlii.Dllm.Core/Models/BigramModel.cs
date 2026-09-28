using EtAlii.Dllm.Core.Numerics;

namespace EtAlii.Dllm.Core.Models;

/// <summary>
/// The smallest possible language model: a table of next-token logits indexed by the previous token, initialised
/// from a seed. It exists to exercise the full deterministic pipeline (tokenizer, forward pass, sampler, API)
/// end to end until the transformer from the roadmap replaces it.
/// </summary>
public sealed class BigramModel : ILanguageModel
{
    private readonly float[] _table;

    public BigramModel(int vocabularySize, ulong seed)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(vocabularySize, 1);
        VocabularySize = vocabularySize;
        Seed = seed;
        _table = new float[vocabularySize * vocabularySize];
        var random = new DeterministicRandom(seed);
        for (var i = 0; i < _table.Length; i++)
        {
            _table[i] = random.NextGaussian();
        }
    }

    public string Id => $"dllm-bigram-{VocabularySize}-{Seed}";

    public int VocabularySize { get; }

    public ulong Seed { get; }

    public void Forward(ReadOnlySpan<int> tokens, Span<float> logits)
    {
        if (logits.Length != VocabularySize)
        {
            throw new ArgumentException("Logits buffer must match the vocabulary size.", nameof(logits));
        }
        var previous = tokens.IsEmpty ? 0 : tokens[^1];
        _table.AsSpan(previous * VocabularySize, VocabularySize).CopyTo(logits);
    }

    /// <summary>Fingerprint of the weights, suitable as an OpenAI style <c>system_fingerprint</c>.</summary>
    public string WeightsFingerprint => Fingerprint.Of(_table);
}
