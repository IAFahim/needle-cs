using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

// MEM-21: prefix snapshot/rollback as a readonly struct token (O(1), no
// recompute). Mark/CheckpointSpec snapshot once into preallocated buffers;
// Rewind(token)/RestoreSpec(token) validate the capture and roll back by
// assignment. These tests pin: token rollback is bit-identical to replay
// from scratch, matches the Export/Import byte round-trip, mirrors the
// parameterless paths exactly, rejects stale/foreign tokens, and allocates
// nothing.
public sealed unsafe class PrefixSnapshotTests : IDisposable
{
    const int PrefixLength = 100;
    readonly Model model;
    readonly ITestOutputHelper output;
    readonly int[] ids;

    public PrefixSnapshotTests(ITestOutputHelper output)
    {
        Assets.Require();
        this.output = output;
        model = Model.Open(Assets.Model);
        ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
    }

    public void Dispose() => model.Dispose();

    float PooledConfidence(Engine engine)
    {
        var pooled = new float[model.Confidence->Queries * model.Geometry.Dim];
        fixed (float* p = pooled) return engine.ConfidenceLogit(p);
    }

    float[] Observe(Engine engine) => engine.Logits.ToArray().Append(PooledConfidence(engine)).ToArray();

    Engine Primed(out Engine.PrefixSnapshot token)
    {
        var engine = new Engine(model, 512, 16) { Pooling = true };
        engine.Feed(ids.AsSpan(0, PrefixLength), computeLogits: false);
        token = engine.Mark();
        return engine;
    }

    // Rollback-bit-equality vs recompute: rewind-to-token plus re-feed must
    // reproduce the exact state of a fresh engine fed the same rows.
    [Fact]
    public void TokenRollbackIsBitIdenticalToReplay()
    {
        using var engine = Primed(out var token);
        Assert.Equal(PrefixLength, token.Length);
        var query = ids.AsSpan(PrefixLength, 24);
        engine.Feed(query);
        var direct = Observe(engine);

        engine.Rewind(token);
        Assert.Equal(PrefixLength, engine.Length);
        engine.Feed(query);
        Assert.Equal(direct, Observe(engine));

        using var replay = new Engine(model, 512, 16) { Pooling = true };
        replay.Feed(ids.AsSpan(0, PrefixLength), computeLogits: false);
        replay.Feed(query);
        Assert.Equal(Observe(replay), Observe(engine));
    }

    // The persisted prefix must carry the same state the token pins:
    // export/import round-trip, then the same query, equals token rollback.
    [Fact]
    public void TokenRollbackMatchesExportImportRoundTrip()
    {
        using var engine = Primed(out var token);
        var query = ids.AsSpan(PrefixLength, 24);
        var bytes = new byte[(int)engine.PrimedPrefixBytes];
        engine.ExportPrimedPrefix(bytes);

        using var restored = new Engine(model, 512, 16) { Pooling = true };
        var importToken = restored.ImportPrimedPrefix(bytes);
        Assert.Equal(PrefixLength, importToken.Length);
        restored.Feed(query);
        var viaBytes = Observe(restored);

        engine.Rewind(token);
        engine.Feed(query);
        Assert.Equal(viaBytes, Observe(engine));
    }

    // Speculative token path mirrors the parameterless checkpoint/restore
    // bit-for-bit (logits and pooled confidence).
    [Fact]
    public void SpecTokenRollbackIsBitExact()
    {
        var prefix = new int[] { 500, 600, 700 };
        var tail = new int[] { 800, 900, 1000, 1100 };
        using var viaToken = new Engine(model, 2048, 64, 1) { Pooling = true };
        using var viaBare = new Engine(model, 2048, 64, 1) { Pooling = true };
        viaToken.Feed(prefix, computeLogits: false);
        viaBare.Feed(prefix, computeLogits: false);
        var spec = viaToken.CheckpointSpec();
        viaBare.CheckpointSpec();
        Assert.Equal(prefix.Length, spec.Length);
        viaToken.Feed(tail, computeLogits: true);
        viaBare.Feed(tail, computeLogits: true);
        viaToken.RestoreSpec(spec);
        viaBare.RestoreSpec();
        Assert.Equal(prefix.Length, viaToken.Length);
        viaToken.Feed(tail[..1], computeLogits: true);
        viaBare.Feed(tail[..1], computeLogits: true);
        Assert.True(viaToken.Logits.ToArray().AsSpan().SequenceEqual(viaBare.Logits.ToArray()), "token-path logits diverged");
        Assert.Equal(PooledConfidence(viaBare), PooledConfidence(viaToken));
    }

    // A re-Mark (prefix edit) retires the earlier token; rolling back to it
    // throws, while the current token still works.
    [Fact]
    public void StalePrefixTokenThrows()
    {
        using var engine = Primed(out var first);
        engine.Feed(ids.AsSpan(PrefixLength, 8));
        var second = engine.Mark();
        Assert.Throws<ArgumentException>(() => engine.Rewind(first));
        engine.Rewind(second);
        Assert.Equal(PrefixLength + 8, engine.Length);
    }

    // Tokens never cross domains: a speculative token is not a prefix
    // rewind point and vice versa. A re-Checkpoint retires the old draft.
    [Fact]
    public void ForeignAndStaleSpecTokensThrow()
    {
        using var engine = Primed(out var prefix);
        var draft = engine.CheckpointSpec();
        Assert.Throws<ArgumentException>(() => engine.Rewind(draft));
        Assert.Throws<ArgumentException>(() => engine.RestoreSpec(prefix));
        engine.Feed(ids.AsSpan(PrefixLength, 4));
        var newer = engine.CheckpointSpec();
        Assert.Throws<ArgumentException>(() => engine.RestoreSpec(draft));
        engine.RestoreSpec(newer);
        Assert.Equal(PrefixLength + 4, engine.Length);
        engine.Rewind(prefix);
        Assert.Equal(PrefixLength, engine.Length);
    }

    // Token capture and rollback allocate nothing (O(1) assignment path).
    [Fact]
    public void TokenRollbackAllocatesNothing()
    {
        using var engine = Primed(out var token);
        engine.Feed(ids.AsSpan(PrefixLength, 8));
        engine.Rewind(token);
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 1000; i++) engine.Rewind(token);
        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
    }
}
