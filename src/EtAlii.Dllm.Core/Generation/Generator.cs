using EtAlii.Dllm.Core.Models;
using EtAlii.Dllm.Core.Numerics;
using EtAlii.Dllm.Core.Sampling;
using EtAlii.Dllm.Core.Tokenization;

namespace EtAlii.Dllm.Core.Generation;

public sealed record GenerationResult(string Text, IReadOnlyList<int> Tokens, int PromptTokens, string FinishReason)
{
    /// <summary>Hash of the generated token ids: equal fingerprints prove bit-identical output.</summary>
    public string Fingerprint => Numerics.Fingerprint.Of(Tokens.ToArray());
}

/// <summary>Runs the autoregressive loop: forward pass, sample, append, repeat.</summary>
public sealed class Generator(ILanguageModel model, ITokenizer tokenizer)
{
    public GenerationResult Generate(string prompt, int maxTokens, SamplingOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(maxTokens);
        var context = new List<int>(tokenizer.Encode(prompt));
        var promptTokens = context.Count;
        var sampler = new Sampler(options);
        var logits = new float[model.VocabularySize];
        var generated = new List<int>(maxTokens);
        var finishReason = "length";

        while (generated.Count < maxTokens)
        {
            model.Forward(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(context), logits);
            var token = sampler.Sample(logits);
            if (token == tokenizer.EndOfSequence)
            {
                finishReason = "stop";
                break;
            }
            generated.Add(token);
            context.Add(token);
        }

        return new GenerationResult(tokenizer.Decode(generated), generated, promptTokens, finishReason);
    }
}
