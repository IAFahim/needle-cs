using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Needle.Cli;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

public sealed class NativeParityTests : IDisposable
{
    const string FixedDate = "date: 2026-10-26 Mon 09:41";
    const int MaxNewTokens = 256;

    readonly Model model;
    readonly ITestOutputHelper output;
    readonly NativeEngine? native;

    public NativeParityTests(ITestOutputHelper output)
    {
        this.output = output;
        Assets.Require();
        model = Model.Open(Assets.Model);
        native = NativeEngine.TryLoad(Assets.Root);
    }

    public void Dispose()
    {
        native?.Dispose();
        model.Dispose();
    }

    bool Ready()
    {
        if (native != null) return true;
        output.WriteLine($"native parity skipped: no upstream wheel in {Path.Combine(Assets.Root, "ref/hf/wheel")}");
        return false;
    }

    // Scalar kernels match AVX2 (and the numpy golden) to cosine 0.9999977 on
    // prefill logits with the same argmax, but that ~1e-6-level realization
    // noise flips near-tied argmaxes and moves head confidence on some
    // prompts (F11 measurement, NEEDLE_SCALAR=1). Bounds below are tier
    // specific; each scalar number cites the measured worst case.
    static bool IsScalar => Tiers.Active == CpuTier.Scalar;

    [Fact]
    public void PrefixMatchesNativeOnEqualSystems()
    {
        if (!Ready()) return;
        foreach (var system in new[] { "", FixedDate })
        {
            var expected = native!.Init(system, File.ReadAllText(Assets.Tools));
            using var session = new NeedleSession(model, File.ReadAllText(Assets.Tools),
                new NeedleOptions { Threads = 4, AutoDate = false, System = system });
            output.WriteLine($"system={(system == "" ? "<empty>" : system)} native={expected} managed={session.ContextUsed}");
            Assert.Equal(expected, session.ContextUsed);
        }
    }

    static readonly (string Query, string System)[] KnifeEdge = new[]
    {
        ("give Lyra 3 potions", ""),
        ("give Lyra 3 potions", FixedDate),
        ("attack the watchtower near the river", ""),
    };

    [Fact]
    public void CallsAndConfidenceMatchNativeOnEqualPrefixes()
    {
        if (!Ready()) return;
        var sets = new[]
        {
            new PromptSet("golden", File.ReadAllText(Assets.Tools), BenchPrompts.Golden),
            new PromptSet("game", BenchPrompts.GameToolsJson, BenchPrompts.Game),
        };
        foreach (var system in new[] { "", FixedDate })
        {
            foreach (var set in sets)
            {
                using (var primed = Fresh(set.ToolsJson, system, 4, 64)) Assert.Equal(native!.Init(system, set.ToolsJson), primed.ContextUsed);
                foreach (var query in set.Queries)
                {
                    if (KnifeEdge.Contains((query, system))) { output.WriteLine($"tie case, see KnifeEdgeTiesDifferFromNative: {query} [{system}]"); continue; }
                    var expected = native.Complete(query, MaxNewTokens);
                    using var session = Fresh(set.ToolsJson, system, 4, 64);
                    var actual = session.Complete(query);
                    Report(query, system, expected, actual);
                    Assert.Equal(expected.CallNames, actual.CallNames());
                    Assert.True(TermsEqual(expected.CallArguments, actual.CallArguments()), $"args differ: native {expected.CallArguments} managed {actual.CallArguments()}");
                    // Scalar worst case 0.1126 ('attack the watchtower near the river' + date system:
                    // native 0.1415, scalar 0.2541 after a tie flip in the reasoning path);
                    // AVX worst case 0.0557 per tools/native_parity_report.md.
                    var confBound = IsScalar ? 0.12 : 0.06;
                    Assert.InRange(actual.Confidence ?? double.NaN, (expected.Confidence ?? 0) - confBound, (expected.Confidence ?? 0) + confBound);
                    Assert.InRange(actual.DecodeConfidence ?? double.NaN, actual.HeadConfidence ?? 1, 1.0);
                }
            }
        }
    }

    [Fact]
    public void KnifeEdgeTiesDifferFromNative()
    {
        if (!Ready()) return;
        var tools = BenchPrompts.GameToolsJson;
        foreach (var system in new[] { "", FixedDate })
        {
            native!.Init(system, tools);
            var lyra = native.Complete("give Lyra 3 potions", MaxNewTokens);
            Assert.Equal("give_item", lyra.CallNames);
            using var first = Fresh(tools, system, 4, 64);
            var said = first.Complete("give Lyra 3 potions");
            // Scalar flips this tie to the native side on the dateless system
            // (measured NEEDLE_SCALAR=1: give_item at conf 0.1854, matching native;
            // dateless AVX stays say at conf 0.1774). The dated system stays say
            // on both tiers (scalar conf 0.1604).
            Assert.Equal(IsScalar && system == "" ? "give_item" : "say", said.CallNames());
            using var second = Fresh(tools, system, 4, 64);
            Assert.Equal(said.CallNames(), second.Complete("give Lyra 3 potions").CallNames());
            if (system == "")
                using (var one = Fresh(tools, system, 1, 64))
                    Assert.Equal(lyra.CallNames, one.Complete("give Lyra 3 potions").CallNames());
        }
        native!.Init("", tools);
        var watchtower = native.Complete("attack the watchtower near the river", MaxNewTokens);
        Assert.Equal("attack", watchtower.CallNames);
        using (var four = Fresh(tools, "", 4, 64))
        {
            var refused = four.Complete("attack the watchtower near the river");
            // Scalar takes the native side here too (measured NEEDLE_SCALAR=1:
            // attack at conf 0.1476 with no suppression; AVX refuses at conf
            // 0.0983 with attack suppressed).
            if (IsScalar)
            {
                Assert.Equal("attack", refused.CallNames());
            }
            else
            {
                Assert.Equal("", refused.CallNames());
                Assert.Equal("attack", refused.SuppressedCalls.CallNames());
            }
        }
        using (var one = Fresh(tools, "", 1, 64))
            Assert.Equal("attack", one.Complete("attack the watchtower near the river").CallNames());
    }

    NeedleSession Fresh(string toolsJson, string system, int threads, int retrievalTopK) =>
        new(model, toolsJson, new NeedleOptions { Threads = threads, AutoDate = false, System = system, RetrievalTopK = retrievalTopK });

    [Fact]
    public void LongContextStaysCloseToNative()
    {
        if (!Ready()) return;
        const string system = FixedDate;
        var tools = File.ReadAllText(Assets.Tools);
        native!.Init(system, tools);
        using var session = new NeedleSession(model, tools,
            new NeedleOptions { Threads = 4, AutoDate = false, System = system, Context = 2048 });
        var turn = new[] { "set a timer for 5 minutes", "what's it like in Lagos right now?", "play some Miles Davis", "set a timer for 3 minutes" };
        var index = 0;
        while (session.ContextUsed < 1500 && index < 64)
        {
            var query = turn[index % turn.Length];
            var expected = native.Continue(query, MaxNewTokens);
            var actual = session.Complete(query);
            // Scalar drifts from native as context grows (both tiers sag while
            // native stays ~0.90; scalar leads AVX by ~100 tokens through the
            // transition, then both sit ~0.75): measured NEEDLE_SCALAR=1 maxima
            // 0.1365 below 1300 tokens (timer3 at tokens~1258) and 0.1538 above
            // (Miles at tokens~1439). AVX maxima are 0.0521/0.1317, hence 0.06/0.15.
            var bound = session.ContextUsed < 1300 ? (IsScalar ? 0.14 : 0.06) : (IsScalar ? 0.16 : 0.15);
            output.WriteLine($"tokens~{session.ContextUsed} q={query} native={expected.Confidence:F4} managed={actual.Confidence:F4} head={actual.HeadConfidence:F4} calls={actual.CallNames()}");
            Assert.Equal(expected.CallNames, actual.CallNames());
            Assert.InRange(actual.Confidence ?? double.NaN, (expected.Confidence ?? 0) - bound, (expected.Confidence ?? 0) + bound);
            index++;
        }
        output.WriteLine($"reached {session.ContextUsed} context tokens in {index} turns");
        Assert.True(session.ContextUsed >= 1500, $"only reached {session.ContextUsed} tokens");
    }

    void Report(string query, string system, NativeResult expected, CallResult actual)
    {
        output.WriteLine($"q={query} sys={(system == "" ? "<empty>" : system)}");
        output.WriteLine($"  native conf={expected.Confidence} calls={expected.CallNames} reasoning={expected.Reasoning}");
        output.WriteLine($"  managed conf={actual.Confidence} head={actual.HeadConfidence} decode={actual.DecodeConfidence} calls={actual.CallNames()} reasoning={actual.Reasoning}");
    }

    static bool TermsEqual(string left, string right)
    {
        var a = left.Length == 0 ? new string[0] : left.Split(';');
        var b = right.Length == 0 ? new string[0] : right.Split(';');
        if (a.Length != b.Length) return false;
        for (var i = 0; i < a.Length; i++)
        {
            using var x = JsonDocument.Parse(a[i]);
            using var y = JsonDocument.Parse(b[i]);
            if (!ValueEqual(x.RootElement, y.RootElement)) return false;
        }
        return true;
    }

    static bool ValueEqual(JsonElement x, JsonElement y)
    {
        if (x.ValueKind != y.ValueKind)
            return x.ValueKind == JsonValueKind.Number && y.ValueKind == JsonValueKind.Number && NumberEqual(x, y);
        return x.ValueKind switch
        {
            JsonValueKind.Number => NumberEqual(x, y),
            JsonValueKind.String => x.GetString() == y.GetString(),
            JsonValueKind.True or JsonValueKind.False or JsonValueKind.Null => true,
            JsonValueKind.Array => x.GetArrayLength() == y.GetArrayLength() && SequenceEqual(x.EnumerateArray(), y.EnumerateArray()),
            _ => ObjectEqual(x, y),
        };
    }

    static bool NumberEqual(JsonElement x, JsonElement y) =>
        Math.Abs(x.GetDouble() - y.GetDouble()) <= 1e-6 * Math.Max(1, Math.Abs(y.GetDouble()));

    static bool SequenceEqual(JsonElement.ArrayEnumerator x, JsonElement.ArrayEnumerator y)
    {
        while (true)
        {
            var nx = x.MoveNext();
            var ny = y.MoveNext();
            if (!nx && !ny) return true;
            if (nx != ny || !ValueEqual(x.Current, y.Current)) return false;
        }
    }

    static bool ObjectEqual(JsonElement x, JsonElement y)
    {
        var count = 0;
        foreach (var property in x.EnumerateObject())
        {
            count++;
            if (!y.TryGetProperty(property.Name, out var other) || !ValueEqual(property.Value, other)) return false;
        }
        foreach (var _ in y.EnumerateObject()) count--;
        return count == 0;
    }

    sealed record NativeResult(string CallNames, string CallArguments, string Reasoning, double? Confidence);

    sealed class NativeEngine : IDisposable
    {
        unsafe delegate int LoadDelegate(byte* cact, ulong n);
        unsafe delegate int InitDelegate(byte* system, byte* tools, byte* index);
        unsafe delegate int CompleteDelegate(byte* input, float* pcm, int samples, int maxNew, byte* @out, int capacity);
        delegate void ResetDelegate();

        readonly IntPtr handle;
        readonly LoadDelegate load;
        readonly InitDelegate init;
        readonly CompleteDelegate complete;
        readonly ResetDelegate reset;
        readonly byte[] cact;
        readonly byte[] output = new byte[1 << 16];

        NativeEngine(IntPtr handle, string library)
        {
            this.handle = handle;
            load = Proc<LoadDelegate>("needle_load");
            init = Proc<InitDelegate>("needle_init");
            complete = Proc<CompleteDelegate>("needle_complete");
            reset = Proc<ResetDelegate>("needle_reset");
            cact = File.ReadAllBytes(Path.Combine(Assets.Root, "ref/hf/needle3.cact"));
            unsafe
            {
                fixed (byte* data = cact)
                    if (load(data, (ulong)cact.Length) < 0) throw new InvalidOperationException($"{library} refused {cact.Length} model bytes");
            }
        }

        T Proc<T>(string name) where T : Delegate =>
            Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(handle, name));

        public static NativeEngine? TryLoad(string root)
        {
            var wheelDir = Path.Combine(root, "ref/hf/wheel");
            string? library = File.Exists(Path.Combine(wheelDir, "libneedle3.so")) ? Path.Combine(wheelDir, "libneedle3.so") : null;
            if (library == null)
            {
                var wheel = Directory.Exists(wheelDir) ? Directory.GetFiles(wheelDir, "*.whl").FirstOrDefault() : null;
                if (wheel == null) return null;
                var target = Path.Combine(Path.GetTempPath(), "needle-cs-native", "libneedle3.so");
                if (!File.Exists(target))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                    using var archive = ZipFile.OpenRead(wheel);
                    var entry = archive.Entries.First(e => e.Name == "libneedle3.so");
                    entry.ExtractToFile(target);
                }
                library = target;
            }
            return new NativeEngine(NativeLibrary.Load(library), library);
        }

        public unsafe int Init(string system, string toolsJson)
        {
            var systemBytes = Encoding.UTF8.GetBytes(system + "\0");
            var toolsBytes = Encoding.UTF8.GetBytes(toolsJson + "\0");
            fixed (byte* s = systemBytes, t = toolsBytes)
                return init(s, t, null);
        }

        public unsafe NativeResult Complete(string query, int maxNewTokens)
        {
            reset();
            return Run(query, maxNewTokens);
        }

        public unsafe NativeResult Continue(string query, int maxNewTokens) => Run(query, maxNewTokens);

        unsafe NativeResult Run(string query, int maxNewTokens)
        {
            var queryBytes = Encoding.UTF8.GetBytes(query + "\0");
            int status;
            fixed (byte* q = queryBytes, o = output)
                status = complete(q, null, 0, maxNewTokens, o, output.Length);
            if (status < 0) throw new InvalidOperationException($"native failed on {query}: {Encoding.UTF8.GetString(output, 0, Array.IndexOf(output, (byte)0))}");
            using var document = JsonDocument.Parse(Encoding.UTF8.GetString(output, 0, Array.IndexOf(output, (byte)0)));
            var root = document.RootElement;
            var calls = root.TryGetProperty("function_calls", out var list) ? list.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
            return new NativeResult(
                string.Join(",", calls.Select(c => c.GetProperty("name").GetString())),
                string.Join(";", calls.Select(c => JsonSerializer.Serialize(c.GetProperty("arguments")))),
                root.TryGetProperty("reasoning", out var reasoning) ? reasoning.GetString() ?? "" : "",
                root.TryGetProperty("confidence", out var confidence) && confidence.ValueKind == JsonValueKind.Number ? confidence.GetDouble() : null);
        }

        public void Dispose() => NativeLibrary.Free(handle);
    }
}

internal static class NativeParityCalls
{
    internal static string CallNames(this CallResult result) => result.FunctionCalls.CallNames();
    internal static string CallNames(this System.Collections.Generic.IEnumerable<FunctionCall> calls) => string.Join(",", calls.Select(c => c.Name));
    internal static string CallArguments(this CallResult result) =>
        string.Join(";", result.FunctionCalls.Select(c => JsonSerializer.Serialize(JsonDocument.Parse(c.ArgumentsJson).RootElement)));
}
