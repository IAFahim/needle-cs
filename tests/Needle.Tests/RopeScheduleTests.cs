using System;
using Xunit;

namespace Needle.Tests;

public sealed unsafe class RopeScheduleTests
{
    [Fact]
    public void FrequenciesAreBitIdenticalToHistoricalFormula()
    {
        const int qkDim = 48;
        const int half = qkDim / 2;
        var freq = stackalloc float[half];
        foreach (var theta in new[] { 10000f, 1000f, 500000f, 10f })
        {
            Engine.RopeFrequencies(theta, qkDim, freq, half);
            for (var i = 0; i < half; i++)
                Assert.Equal(1f / MathF.Pow(theta, 2f * i / qkDim), freq[i]);
        }
    }
}
