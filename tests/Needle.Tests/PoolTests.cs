using System.Runtime.InteropServices;
using Xunit;

namespace Needle.Tests;

public sealed unsafe class PoolTests
{
    const int Slots = 128;

    static void Increment(void* context, int index) => ((int*)context)[index] += 1;

    [Fact]
    public void EveryTaskRunsExactlyOnceAcrossManyBackToBackJobs()
    {
        using var pool = new Pool(8);
        var counts = (int*)NativeMemory.AllocZeroed(Slots * sizeof(int));
        var expected = new int[Slots];
        var mismatches = 0;
        try
        {
            for (var job = 0; job < 50_000; job++)
            {
                var tasks = 1 + job % 97;
                pool.Run(&Increment, counts, tasks);
                for (var i = 0; i < tasks; i++) expected[i]++;
                for (var i = 0; i < Slots; i++) if (counts[i] != expected[i]) mismatches++;
            }
        }
        finally { NativeMemory.Free(counts); }
        Assert.Equal(0, mismatches);
    }
}
