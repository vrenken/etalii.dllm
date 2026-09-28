namespace EtAlii.Dllm.Core.Models;

/// <summary>
/// An autoregressive language model. Implementations must be pure: the logits depend only on the weights and
/// the given tokens, and are bit-identical on every platform.
/// </summary>
public interface ILanguageModel
{
    string Id { get; }

    int VocabularySize { get; }

    /// <summary>Writes the next-token logits for the given context into <paramref name="logits"/>.</summary>
    void Forward(ReadOnlySpan<int> tokens, Span<float> logits);
}
