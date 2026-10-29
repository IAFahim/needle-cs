using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;

namespace Needle.Tests;

public sealed class LoadingTests
{
    sealed class ForwardOnlyStream(Stream inner) : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, Math.Min(count, 100_003));
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    static float[] Case0Logits(Model model)
    {
        var ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
        using var engine = new Engine(model, 512, 64, 1);
        engine.Feed(ids);
        return engine.Logits.ToArray();
    }

    static float[] ExpectedCase0Logits() => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/logits.bin"))).ToArray();

    static void AssertMatchesReference(float[] actual)
    {
        var expected = ExpectedCase0Logits();
        Assert.Equal(Array.IndexOf(expected, expected.Max()), Array.IndexOf(actual, actual.Max()));
        Assert.True(Cosine(actual, expected) > 0.99995, $"cosine {Cosine(actual, expected)}");
    }

    static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / Math.Sqrt(na * nb);
    }

    [Fact]
    public unsafe void EverySourceProducesIdenticalLogits()
    {
        Assets.Require();
        var bytes = File.ReadAllBytes(Assets.Model);

        float[] fromFile;
        using (var model = Model.Open(Assets.Model)) fromFile = Case0Logits(model);
        AssertMatchesReference(fromFile);

        float[] fromBytes;
        using (var model = Model.FromBytes(bytes)) fromBytes = Case0Logits(model);

        float[] fromMemory;
        fixed (byte* pinned = bytes)
        using (var model = Model.FromMemory(pinned, (nuint)bytes.Length)) fromMemory = Case0Logits(model);

        float[] fromStream;
        using (var model = Model.FromStream(new MemoryStream(bytes))) fromStream = Case0Logits(model);

        float[] fromForwardOnlyStream;
        using (var model = Model.FromStream(new ForwardOnlyStream(new MemoryStream(bytes)))) fromForwardOnlyStream = Case0Logits(model);

        Assert.Equal(fromFile, fromBytes);
        Assert.Equal(fromFile, fromMemory);
        Assert.Equal(fromFile, fromStream);
        Assert.Equal(fromFile, fromForwardOnlyStream);
    }

    [Fact]
    public void LoadingTwiceAndDisposingKeepsEachModelUsable()
    {
        Assets.Require();
        var first = Model.Open(Assets.Model);
        var second = Model.Open(Assets.Model, prefault: false);
        var expected = Case0Logits(first);
        first.Dispose();
        first.Dispose();
        Assert.Equal(expected, Case0Logits(second));
        second.Prefault();
        second.Dispose();
        second.Dispose();
        using var third = Model.Open(Assets.Model);
        AssertMatchesReference(Case0Logits(third));
    }

    [Fact]
    public void RepeatedOpenAndDisposeIsStable()
    {
        Assets.Require();
        for (var i = 0; i < 20; i++)
        {
            using var model = Model.Open(Assets.Model, prefault: i % 2 == 0);
            Assert.True(model.LayerCount > 0);
        }
    }

    [Fact]
    public void MappedFileCanBeOpenedAgainWhileAModelHoldsIt()
    {
        Assets.Require();
        using var model = Model.Open(Assets.Model);
        using var stream = File.OpenRead(Assets.Model);
        Assert.True(stream.Length > 0);
    }

    [Fact]
    public void MissingPathThrowsFileNotFound()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "missing.cact");
        Assert.ThrowsAny<IOException>(() => Model.Open(path));
        Assert.Throws<FileNotFoundException>(() => Model.Open(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".cact")));
    }

    [Fact]
    public void EmptyAndGarbageFilesThrowInvalidData()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".cact");
        try
        {
            File.WriteAllBytes(path, Array.Empty<byte>());
            Assert.Throws<InvalidDataException>(() => Model.Open(path));
            File.WriteAllBytes(path, new byte[4096]);
            Assert.Throws<InvalidDataException>(() => Model.Open(path));
            Assert.Throws<InvalidDataException>(() => Model.FromBytes(new byte[4096]));
            Assert.Throws<InvalidDataException>(() => Model.FromBytes(Array.Empty<byte>()));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
