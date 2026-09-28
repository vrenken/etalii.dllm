using System.ComponentModel;
using EtAlii.Dllm.Core.Hosting;
using EtAlii.Dllm.Core.Sampling;
using ModelContextProtocol.Server;

namespace EtAlii.Dllm.Mcp;

[McpServerToolType]
public sealed class DllmTools(DllmEngine engine)
{
    [McpServerTool(Name = "generate", ReadOnly = true, Idempotent = true)]
    [Description("Generates a continuation of the prompt with the EtAlii deterministic LLM. The same arguments always return the same text.")]
    public string Generate(
        [Description("The text to continue.")] string prompt,
        [Description("Maximum number of tokens to generate.")] int maxTokens = 64,
        [Description("Sampling temperature; 0 means greedy.")] float temperature = 0f,
        [Description("Seed for sampling when temperature > 0.")] long seed = 0)
    {
        var result = engine.Complete(prompt, maxTokens, new SamplingOptions(Temperature: temperature, Seed: unchecked((ulong)seed)));
        return result.Text;
    }

    [McpServerTool(Name = "model_info", ReadOnly = true, Idempotent = true)]
    [Description("Returns the model id and the system fingerprint that identifies the exact weights.")]
    public string ModelInfo() => $"model: {engine.Model.Id}\nsystem_fingerprint: {engine.SystemFingerprint}";
}
