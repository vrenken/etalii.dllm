using System.Text.Json;
using EtAlii.Dllm.Core.Chat;
using EtAlii.Dllm.Core.Hosting;
using EtAlii.Dllm.Core.Sampling;
using EtAlii.Dllm.Server;

var builder = WebApplication.CreateBuilder(args);
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
});
builder.Services.AddSingleton(DllmEngine.CreateDefault());

var app = builder.Build();

app.MapGet("/v1/models", (DllmEngine engine) =>
    new ModelList("list", [new ModelInfo(engine.Model.Id, "model", 0, "etalii")]));

app.MapPost("/v1/chat/completions", (ChatCompletionRequest request, DllmEngine engine) =>
{
    if (request.Stream == true)
    {
        return Results.BadRequest(new ErrorResponse(new ErrorBody("Streaming is not supported yet.", "invalid_request_error")));
    }
    if (request.Messages.Count == 0)
    {
        return Results.BadRequest(new ErrorResponse(new ErrorBody("'messages' must contain at least one message.", "invalid_request_error")));
    }

    var options = new SamplingOptions(
        Temperature: request.Temperature ?? 0f,
        TopP: request.TopP ?? 1f,
        Seed: unchecked((ulong)(request.Seed ?? 0)));
    var maxTokens = request.MaxCompletionTokens ?? request.MaxTokens ?? 64;
    var result = engine.Chat(request.Messages.Select(m => new ChatMessage(m.Role, m.Content ?? string.Empty)), maxTokens, options);

    // The id and timestamp are derived from the output, not from the clock, so identical requests give identical responses.
    return Results.Ok(new ChatCompletionResponse(
        Id: "chatcmpl-" + result.Fingerprint[..24],
        Object: "chat.completion",
        Created: 0,
        Model: engine.Model.Id,
        SystemFingerprint: engine.SystemFingerprint,
        Choices: [new ChatCompletionChoice(0, new ChatCompletionMessage("assistant", result.Text), result.FinishReason)],
        Usage: new ChatCompletionUsage(result.PromptTokens, result.Tokens.Count, result.PromptTokens + result.Tokens.Count)));
});

app.Run();

public partial class Program;
