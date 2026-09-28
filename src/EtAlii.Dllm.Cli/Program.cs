using System.Globalization;
using EtAlii.Dllm.Core.Hosting;
using EtAlii.Dllm.Core.Sampling;

// dllm generate --prompt <text> [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--seed S]
// dllm info

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine("""
        dllm - EtAlii deterministic LLM

        Commands:
          generate --prompt <text> [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--seed S]
          info
        """);
    return 0;
}

var engine = DllmEngine.CreateDefault();

switch (args[0])
{
    case "info":
        Console.WriteLine($"model:              {engine.Model.Id}");
        Console.WriteLine($"system_fingerprint: {engine.SystemFingerprint}");
        return 0;

    case "generate":
        var options = ParseOptions(args[1..]);
        var prompt = options.GetValueOrDefault("prompt") ?? string.Empty;
        var sampling = new SamplingOptions(
            Temperature: float.Parse(options.GetValueOrDefault("temperature") ?? "0", CultureInfo.InvariantCulture),
            TopK: int.Parse(options.GetValueOrDefault("top-k") ?? "0", CultureInfo.InvariantCulture),
            TopP: float.Parse(options.GetValueOrDefault("top-p") ?? "1", CultureInfo.InvariantCulture),
            Seed: ulong.Parse(options.GetValueOrDefault("seed") ?? "0", CultureInfo.InvariantCulture));
        var maxTokens = int.Parse(options.GetValueOrDefault("max-tokens") ?? "64", CultureInfo.InvariantCulture);
        var result = engine.Complete(prompt, maxTokens, sampling);
        Console.WriteLine(result.Text);
        Console.Error.WriteLine($"fingerprint: {result.Fingerprint}  tokens: {result.Tokens.Count}  finish: {result.FinishReason}");
        return 0;

    default:
        Console.Error.WriteLine($"Unknown command '{args[0]}'. Run 'dllm --help'.");
        return 1;
}

static Dictionary<string, string> ParseOptions(string[] args)
{
    var options = new Dictionary<string, string>(StringComparer.Ordinal);
    for (var i = 0; i + 1 < args.Length; i += 2)
    {
        if (!args[i].StartsWith("--", StringComparison.Ordinal))
        {
            throw new ArgumentException($"Expected an option but got '{args[i]}'.");
        }
        options[args[i][2..]] = args[i + 1];
    }
    return options;
}
