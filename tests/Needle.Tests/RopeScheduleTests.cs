using System;
using Xunit;

namespace Needle.Tests;

// MATH-M7: hoisted RoPE frequency schedule (Engine.RopeFrequencies).
// The schedule is deliberately bit-identical to the historical per-entry
// formula, not a log-domain exp reformulation: ~1e-7 exp-vs-Pow noise flips
// the documented 'attack the watchtower near the river' knife-edge between
// native-refuse and managed-call (NativeParityTests). This pins that contract.
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
