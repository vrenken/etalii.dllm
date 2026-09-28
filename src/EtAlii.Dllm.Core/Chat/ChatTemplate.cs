using System.Text;

namespace EtAlii.Dllm.Core.Chat;

public sealed record ChatMessage(string Role, string Content);

/// <summary>Flattens a chat conversation into a single prompt, in a fixed, documented format.</summary>
public static class ChatTemplate
{
    public static string Render(IEnumerable<ChatMessage> messages)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var builder = new StringBuilder();
        foreach (var message in messages)
        {
            builder.Append("<|").Append(message.Role).Append("|>\n").Append(message.Content).Append('\n');
        }
        return builder.Append("<|assistant|>\n").ToString();
    }
}
