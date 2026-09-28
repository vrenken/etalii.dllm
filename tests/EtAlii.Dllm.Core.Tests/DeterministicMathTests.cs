using EtAlii.Dllm.Core.Numerics;

namespace EtAlii.Dllm.Core.Tests;

public class DeterministicMathTests
{
    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    [InlineData(0.5)]
    [InlineData(10.0)]
    [InlineData(-20.0)]
    [InlineData(88.7)]
    [InlineData(-103.9)]
    [InlineData(700.0)]
    public void Exp_IsAccurate(double x)
    {
        var expected = Math.Exp(x);
        var actual = DeterministicMath.Exp(x);
        Assert.True(Math.Abs(actual - expected) <= Math.Abs(expected) * 1e-14, $"exp({x}) = {actual}, expected {expected}");
    }

    [Fact]
    public void Exp_HandlesEdgeCases()
    {
        Assert.Equal(1.0, DeterministicMath.Exp(0.0));
        Assert.Equal(double.PositiveInfinity, DeterministicMath.Exp(1000.0));
        Assert.Equal(0.0, DeterministicMath.Exp(-1000.0));
        Assert.True(double.IsNaN(DeterministicMath.Exp(double.NaN)));
    }

    [Fact]
    public void Softmax_SumsToOne()
    {
        float[] logits = [1f, 2f, 3f, -4f, 0.5f];
        var probabilities = new float[logits.Length];
        DeterministicMath.Softmax(logits, probabilities);
        Assert.Equal(1.0, DeterministicMath.Sum(probabilities), 6);
        Assert.Equal(2, DeterministicMath.ArgMax(probabilities));
    }

    [Fact]
    public void ArgMax_BreaksTiesTowardsLowestIndex()
    {
        Assert.Equal(1, DeterministicMath.ArgMax([0f, 5f, 5f, 1f]));
    }

    [Fact]
    public void Random_SequenceIsStable()
    {
        // Golden values: if these change, every seeded result in the project changes with them.
        var random = new DeterministicRandom(42);
        Assert.Equal(GoldenValues.RandomFirst, random.NextUInt64());
        Assert.Equal(GoldenValues.RandomSecond, random.NextUInt64());
    }
}
