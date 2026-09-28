using EtAlii.Dllm.Core.Chat;
using EtAlii.Dllm.Core.Generation;
using EtAlii.Dllm.Core.Models;
using EtAlii.Dllm.Core.Sampling;
using EtAlii.Dllm.Core.Tokenization;

namespace EtAlii.Dllm.Core.Hosting;

/// <summary>
/// Facade shared by the CLI, the OpenAI-compatible server and the MCP server so all three produce identical output.
/// </summary>
public sealed class DllmEngine
{
    public const ulong DefaultModelSeed = 42;

    private readonly ITokenizer _tokenizer;
    private readonly Generator _generator;

    public DllmEngine(ILanguageModel model, ITokenizer tokenizer, string systemFingerprint)
    {
        Model = model;
        _tokenizer = tokenizer;
        SystemFingerprint = systemFingerprint;
        _generator = new Generator(model, tokenizer);
    }

    public ILanguageModel Model { get; }

    /// <summary>Identifies the exact weights and engine version; equal fingerprints plus equal requests give equal output.</summary>
    public string SystemFingerprint { get; }

    public static DllmEngine CreateDefault()
    {
        var tokenizer = new ByteTokenizer();
        var model = new BigramModel(tokenizer.VocabularySize, DefaultModelSeed);
        return new DllmEngine(model, tokenizer, "fp_" + model.WeightsFingerprint[..12]);
    }

    public GenerationResult Complete(string prompt, int maxTokens, SamplingOptions options) =>
        _generator.Generate(prompt, maxTokens, options);

    public GenerationResult Chat(IEnumerable<ChatMessage> messages, int maxTokens, SamplingOptions options) =>
        Complete(ChatTemplate.Render(messages), maxTokens, options);
}
