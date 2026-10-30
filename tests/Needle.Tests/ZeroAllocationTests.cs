using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Xunit;

namespace Needle.Tests;

internal static unsafe class MallocProbe
{
    static readonly nint Counter = Find();

    public static bool Available => Counter != 0;

    public static long Count => Counter == 0 ? 0 : ((delegate* unmanaged<long>)Counter)();

    static nint Find() =>
        NativeLibrary.TryGetExport(NativeLibrary.GetMainProgramHandle(), "needle_malloc_count", out var address) ? address : 0;
}

internal static class AllocationProbe
{
    public const string Variable = "NEEDLE_ALLOCATION_PROBE";
    const int WarmUpRounds = 20;
    const int MeasuredRounds = 100;

    public readonly record struct Measurement(long CallerThread, long WholeProcess, long NativeCalls, bool NativeCounted, int Unexpected);

    public static Measurement Measure(string mode, int threads)
    {
        Assets.Require();
        using var model = Model.Open(Assets.Model);
        var retrieval = mode == "retrieval";
        var tools = retrieval ? SessionCases.Catalog() : SessionCases.DefaultTools();
        // The jumpforward mode decodes with forced-span chunking on the same
        // prompts: the chunk buffers are stack-owned, so the 0 B bar holds.
        using var session = new NeedleSession(model, tools, new NeedleOptions { Threads = threads, AutoDate = false, RetrievalTopK = 5, JumpForward = mode == "jumpforward" });
        var calls = new byte[4096];
        var reasoning = new byte[4096];
        var response = new byte[4096];
        for (var round = 0; round < WarmUpRounds; round++) Round(session, retrieval, round, calls, reasoning, response);

        var unexpected = 0;
        var nativeBefore = MallocProbe.Count;
        var threadBefore = GC.GetAllocatedBytesForCurrentThread();
        var processBefore = GC.GetTotalAllocatedBytes(true);
        for (var round = 0; round < MeasuredRounds; round++)
            if (Round(session, retrieval, round, calls, reasoning, response) == ReplyStatus.Failed) unexpected++;
        var processAfter = GC.GetTotalAllocatedBytes(true);
        var threadAfter = GC.GetAllocatedBytesForCurrentThread();
        var nativeAfter = MallocProbe.Count;
        return new Measurement(threadAfter - threadBefore, processAfter - processBefore, nativeAfter - nativeBefore, MallocProbe.Available, unexpected);
    }

    static ReplyStatus Round(NeedleSession session, bool retrieval, int round, byte[] calls, byte[] reasoning, byte[] response)
    {
        session.Reset();
        if (retrieval) return session.Complete("set an alarm for seven in the morning"u8, calls, reasoning, response).Status;
        var reply = (round & 1) == 0
            ? session.Complete("what's it like in Lagos right now?"u8, calls, reasoning, response)
            : session.Complete("set a timer for 12 minutes and play some Miles Davis"u8, calls, reasoning, response);
        if ((round & 3) == 0 && reply.Status == ReplyStatus.Called)
            reply = session.CompleteWithToolResult("{\"city\":\"Lagos\",\"temp_c\":27,\"sky\":\"clear\"}"u8, calls, reasoning, response);
        return reply.Status;
    }
}

internal static class AllocationProbeEntry
{
#pragma warning disable CA2255
    [ModuleInitializer]
    internal static void Run()
#pragma warning restore CA2255
    {
        var request = Environment.GetEnvironmentVariable(AllocationProbe.Variable);
        if (request == null) return;
        var parts = request.Split(':');
        var measurement = AllocationProbe.Measure(parts[0], int.Parse(parts[1]));
        Console.Out.WriteLine($"probe caller={measurement.CallerThread} process={measurement.WholeProcess} native={measurement.NativeCalls} nativeCounted={measurement.NativeCounted} unexpected={measurement.Unexpected}");
        Console.Out.Flush();
        Environment.Exit(0);
    }
}

public sealed class ZeroAllocationTests
{
    static AllocationProbe.Measurement RunInQuietProcess(string mode, int threads)
    {
        Assets.Require();
        var host = Environment.ProcessPath is { } path && Path.GetFileNameWithoutExtension(path) == "dotnet" ? path : Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") ?? "dotnet";
        var assembly = typeof(ZeroAllocationTests).Assembly.Location;
        var stem = Path.Combine(Path.GetDirectoryName(assembly)!, Path.GetFileNameWithoutExtension(assembly));
        var start = new ProcessStartInfo(host) { RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var argument in new[] { "exec", "--runtimeconfig", stem + ".runtimeconfig.json", "--depsfile", stem + ".deps.json", assembly }) start.ArgumentList.Add(argument);
        start.Environment[AllocationProbe.Variable] = $"{mode}:{threads}";
        using var process = Process.Start(start)!;
        var error = process.StandardError.ReadToEndAsync();
        var output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        var line = output.Split('\n', StringSplitOptions.RemoveEmptyEntries)[^1].Trim();
        Assert.True(line.StartsWith("probe ", StringComparison.Ordinal), $"{line} {error.Result}");
        var fields = line[6..].Split(' ');
        long Field(int index) => long.Parse(fields[index][(fields[index].IndexOf('=') + 1)..]);
        return new AllocationProbe.Measurement(Field(0), Field(1), Field(2), fields[3].EndsWith("True", StringComparison.Ordinal), (int)Field(4));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void HundredWarmSpanCallsAllocateNothingManagedOrNative(int threads)
    {
        var measurement = RunInQuietProcess("plain", threads);
        Assert.Equal(0, measurement.Unexpected);
        Assert.Equal(0, measurement.CallerThread);
        Assert.Equal(0, measurement.WholeProcess);
        if (measurement.NativeCounted) Assert.Equal(0, measurement.NativeCalls);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void HundredWarmJumpForwardCallsAllocateNothingManagedOrNative(int threads)
    {
        var measurement = RunInQuietProcess("jumpforward", threads);
        Assert.Equal(0, measurement.Unexpected);
        Assert.Equal(0, measurement.CallerThread);
        Assert.Equal(0, measurement.WholeProcess);
        if (measurement.NativeCounted) Assert.Equal(0, measurement.NativeCalls);
    }

    [Fact]
    public void HundredWarmSpanCallsWithToolRetrievalAllocateNothingManagedOrNative()
    {
        var measurement = RunInQuietProcess("retrieval", 1);
        Assert.Equal(0, measurement.Unexpected);
        Assert.Equal(0, measurement.CallerThread);
        Assert.Equal(0, measurement.WholeProcess);
        if (measurement.NativeCounted) Assert.Equal(0, measurement.NativeCalls);
    }

    [Fact]
    public void StringWrapperAllocatesOnlyTheManagedResult()
    {
        Assets.Require();
        using var model = Model.Open(Assets.Model);
        using var session = new NeedleSession(model, SessionCases.DefaultTools(), new NeedleOptions { Threads = 1, AutoDate = false });
        for (var round = 0; round < 3; round++)
        {
            session.Reset();
            session.Complete("what's it like in Lagos right now?");
        }
        session.Reset();
        var before = GC.GetAllocatedBytesForCurrentThread();
        var result = session.Complete("what's it like in Lagos right now?");
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(result.Success);
        Assert.InRange(allocated, 1, 4096);
    }
}
