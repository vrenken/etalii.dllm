using EtAlii.Dllm.Core.Numerics;

namespace EtAlii.Dllm.Core.Sampling;

/// <summary>
/// Deterministic token sampler. Candidates are ordered by (probability descending, token id ascending),
/// a total order, so ties never depend on sort stability or hardware.
/// </summary>
public sealed class Sampler
{
    private readonly SamplingOptions _options;
    private DeterministicRandom _random;

    public Sampler(SamplingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.Temperature < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "Temperature must be non-negative.");
        }
        if (options.TopP is <= 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "TopP must be in (0, 1].");
        }
        _options = options;
        _random = new DeterministicRandom(options.Seed);
    }

    public int Sample(ReadOnlySpan<float> logits)
    {
        if (_options.Temperature == 0f)
        {
            return DeterministicMath.ArgMax(logits);
        }

        var scaled = new float[logits.Length];
        for (var i = 0; i < logits.Length; i++)
        {
            scaled[i] = logits[i] / _options.Temperature;
        }
        var probabilities = new float[logits.Length];
        DeterministicMath.Softmax(scaled, probabilities);

        var order = new int[logits.Length];
        for (var i = 0; i < order.Length; i++)
        {
            order[i] = i;
        }
        Array.Sort(order, (a, b) =>
        {
            var byProbability = probabilities[b].CompareTo(probabilities[a]);
            return byProbability != 0 ? byProbability : a.CompareTo(b);
        });

        var keep = order.Length;
        if (_options.TopK > 0)
        {
            keep = Math.Min(keep, _options.TopK);
        }
        if (_options.TopP < 1f)
        {
            var cumulative = 0.0;
            for (var i = 0; i < keep; i++)
            {
                cumulative += probabilities[order[i]];
                if (cumulative >= _options.TopP)
                {
                    keep = i + 1;
                    break;
                }
            }
        }

        var total = 0.0;
        for (var i = 0; i < keep; i++)
        {
            total += probabilities[order[i]];
        }
        var target = _random.NextDouble() * total;
        var running = 0.0;
        for (var i = 0; i < keep; i++)
        {
            running += probabilities[order[i]];
            if (target < running)
            {
                return order[i];
            }
        }
        return order[keep - 1];
    }
}
