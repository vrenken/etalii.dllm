using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;

namespace EtAlii.Dllm.Server.Tests;

public class ChatCompletionsTests(WebApplicationFactory<Program> factory) : IClassFixture<WebApplicationFactory<Program>>
{
    private static readonly object Request = new
    {
        model = "dllm-bigram-257-42",
        messages = new[] { new { role = "user", content = "Say something deterministic." } },
        temperature = 0.7,
        seed = 5,
        max_tokens = 24,
    };

    [Fact]
    public async Task ListsModels()
    {
        using var client = factory.CreateClient();
        var json = await client.GetFromJsonAsync<JsonElement>("/v1/models");
        Assert.Equal("list", json.GetProperty("object").GetString());
        Assert.Equal("dllm-bigram-257-42", json.GetProperty("data")[0].GetProperty("id").GetString());
    }

    [Fact]
    public async Task IdenticalRequests_ReturnIdenticalResponses()
    {
        using var client = factory.CreateClient();
        var first = await (await client.PostAsJsonAsync("/v1/chat/completions", Request)).Content.ReadAsStringAsync();
        var second = await (await client.PostAsJsonAsync("/v1/chat/completions", Request)).Content.ReadAsStringAsync();
        Assert.Equal(first, second);

        var json = JsonDocument.Parse(first).RootElement;
        Assert.Equal("chat.completion", json.GetProperty("object").GetString());
        Assert.StartsWith("fp_", json.GetProperty("system_fingerprint").GetString(), StringComparison.Ordinal);
        Assert.Equal("assistant", json.GetProperty("choices")[0].GetProperty("message").GetProperty("role").GetString());
        Assert.Equal(24, json.GetProperty("usage").GetProperty("completion_tokens").GetInt32());
    }

    [Fact]
    public async Task RejectsEmptyMessages()
    {
        using var client = factory.CreateClient();
        var response = await client.PostAsJsonAsync("/v1/chat/completions", new { messages = Array.Empty<object>() });
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}
