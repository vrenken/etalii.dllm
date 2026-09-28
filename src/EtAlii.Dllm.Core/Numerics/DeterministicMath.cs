namespace EtAlii.Dllm.Core.Numerics;

/// <summary>
/// Floating point kernels whose results are bit-exact on every platform.
/// </summary>
/// <remarks>
/// Rules followed by every method here (see docs/research/deterministic-inference.md):
/// <list type="bullet">
/// <item>Only IEEE 754 basic operations (+, -, *, /, sqrt), which are correctly rounded and therefore portable.</item>
/// <item>No calls into <see cref="Math"/> or <see cref="MathF"/> transcendental functions: those delegate to the
/// platform C runtime and may differ in the last bit between Windows, Linux, macOS, x64 and Arm64.</item>
/// <item>Reductions run in one fixed, sequential order. No hardware-width-dependent SIMD reductions, no parallelism.</item>
/// <item>Accumulation happens in double precision; the single rounding back to float is part of the contract.</item>
/// </list>
/// </remarks>
public static class DeterministicMath
{
    private const double Ln2Hi = 6.93147180369123816490e-01;
    private const double Ln2Lo = 1.90821492927058770002e-10;
    private const double InvLn2 = 1.44269504088896338700e+00;

    /// <summary>Sums the values in index order with a double precision accumulator.</summary>
    public static double Sum(ReadOnlySpan<float> values)
    {
        var sum = 0.0;
        foreach (var v in values)
        {
            sum += v;
        }
        return sum;
    }

    /// <summary>Dot product in index order with a double precision accumulator.</summary>
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vectors must have the same length.", nameof(b));
        }
        var sum = 0.0;
        for (var i = 0; i < a.Length; i++)
        {
            sum += (double)a[i] * b[i];
        }
        return (float)sum;
    }

    /// <summary>Returns the index of the largest value; ties resolve to the lowest index.</summary>
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        if (values.IsEmpty)
        {
            throw new ArgumentException("Cannot take the arg max of an empty span.", nameof(values));
        }
        var best = 0;
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// Portable e^x. Range reduction x = k·ln2 + r with a two-part ln2, then a fixed degree-13 Taylor
    /// polynomial for e^r evaluated with Horner's scheme, then an exact scaling by 2^k.
    /// Accurate to within a few ulps of double precision, and identical everywhere.
    /// </summary>
    public static double Exp(double x)
    {
        if (double.IsNaN(x))
        {
            return x;
        }
        if (x > 709.78)
        {
            return double.PositiveInfinity;
        }
        if (x < -745.2)
        {
            return 0.0;
        }

        // Round to nearest without Math.Round's mode dependence: truncate after offsetting by ±0.5.
        var kd = x * InvLn2;
        var k = (int)(kd >= 0 ? kd + 0.5 : kd - 0.5);
        var r = (x - k * Ln2Hi) - k * Ln2Lo;

        // |r| <= ln2/2, so 13 terms put the truncation error below double epsilon.
        var p = 1.0 / 6227020800.0;
        p = p * r + 1.0 / 479001600.0;
        p = p * r + 1.0 / 39916800.0;
        p = p * r + 1.0 / 3628800.0;
        p = p * r + 1.0 / 362880.0;
        p = p * r + 1.0 / 40320.0;
        p = p * r + 1.0 / 5040.0;
        p = p * r + 1.0 / 720.0;
        p = p * r + 1.0 / 120.0;
        p = p * r + 1.0 / 24.0;
        p = p * r + 1.0 / 6.0;
        p = p * r + 0.5;
        p = p * r + 1.0;
        p = p * r + 1.0;

        return ScaleByPowerOfTwo(p, k);
    }

    /// <summary>
    /// Numerically stable softmax: subtract the maximum, exponentiate with <see cref="Exp"/>,
    /// normalise by the sequentially accumulated sum.
    /// </summary>
    public static void Softmax(ReadOnlySpan<float> logits, Span<float> probabilities)
    {
        if (probabilities.Length != logits.Length)
        {
            throw new ArgumentException("Output must have the same length as the input.", nameof(probabilities));
        }
        var max = logits[ArgMax(logits)];
        var sum = 0.0;
        for (var i = 0; i < logits.Length; i++)
        {
            var e = Exp((double)logits[i] - max);
            probabilities[i] = (float)e;
            sum += e;
        }
        var inv = 1.0 / sum;
        for (var i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] = (float)(probabilities[i] * inv);
        }
    }

    /// <summary>Multiplies by 2^k exactly (for results in the normal range) by building the power of two from bits.</summary>
    private static double ScaleByPowerOfTwo(double value, int k)
    {
        // Split large exponents so each factor stays a normal double.
        while (k > 1023)
        {
            value *= BitConverter.Int64BitsToDouble(0x7FE0000000000000L);
            k -= 1023;
        }
        while (k < -1022)
        {
            value *= BitConverter.Int64BitsToDouble(0x0010000000000000L);
            k += 1022;
        }
        return value * BitConverter.Int64BitsToDouble((long)(k + 1023) << 52);
    }
}
