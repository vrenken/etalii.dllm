namespace EtAlii.Dllm.Server;

#pragma warning disable CA1720 // "object" is part of the OpenAI wire format.

// Subset of the OpenAI Chat Completions wire format. Property names are mapped to snake_case by the JSON options.

public sealed record ChatCompletionRequest
{
    public string? Model { get; init; }

    public List<ChatCompletionMessage> Messages { get; init; } = [];

    public float? Temperature { get; init; }

    public float? TopP { get; init; }

    /// <summary>OpenAI documents <c>seed</c> as best effort; here it is a guarantee.</summary>
    public long? Seed { get; init; }

    public int? MaxTokens { get; init; }

    public int? MaxCompletionTokens { get; init; }

    public bool? Stream { get; init; }
}

public sealed record ChatCompletionMessage(string Role, string? Content);

public sealed record ChatCompletionResponse(
    string Id,
    string Object,
    long Created,
    string Model,
    string SystemFingerprint,
    IReadOnlyList<ChatCompletionChoice> Choices,
    ChatCompletionUsage Usage);

public sealed record ChatCompletionChoice(int Index, ChatCompletionMessage Message, string FinishReason);

public sealed record ChatCompletionUsage(int PromptTokens, int CompletionTokens, int TotalTokens);

public sealed record ModelList(string Object, IReadOnlyList<ModelInfo> Data);

public sealed record ModelInfo(string Id, string Object, long Created, string OwnedBy);

public sealed record ErrorResponse(ErrorBody Error);

public sealed record ErrorBody(string Message, string Type);
