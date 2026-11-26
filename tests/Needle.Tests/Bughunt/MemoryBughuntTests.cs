using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Xunit;

namespace Needle.Tests.Bughunt;

internal sealed class SyntheticCact
{
    public const byte Fp16 = 1, Fp32 = 2, Cq = 3, Raw = 4;
    public const int RecordBytes = 44;

    public sealed class Tensor
    {
        public byte Dtype;
        public uint[] Shape = Array.Empty<uint>();
        public byte[] Data = Array.Empty<byte>();
        public uint Bits;
        public uint Group;
        public long? OffsetOverride;
        public long? BytesOverride;
    }

    public readonly uint[] Header = new uint[49];
    public readonly List<Tensor> Tensors = new();
    public int CodebookEntries = 28;

    public SyntheticCact(int layers = 0, int dim = 128, int vocab = 4, int hadaN = 64)
    {
        Header[0] = ModelInfo.SupportedMagic;
        Header[3] = 64;
        Header[5] = (uint)vocab;
        Header[7] = (uint)dim;
        Header[8] = 2;
        Header[9] = 1;
        Header[10] = (uint)layers;
        Header[11] = 48;
        Header[12] = 64;
        Header[13] = 64;
        Header[14] = (uint)hadaN;
        Header[15] = 4;
        Header[16] = 64;
        Header[19] = 3;
        Header[23] = 4;
        Header[26] = 0;
        Header[31] = 0;
        BinaryPrimitives.WriteSingleLittleEndian(MemoryMarshal.AsBytes(Header.AsSpan(48, 1)), 10000f);
    }

    public static Tensor Dense(params uint[] shape)
    {
        long count = 1;
        foreach (var s in shape) count *= s;
        return new Tensor { Dtype = Fp32, Shape = shape, Data = new byte[count * 4] };
    }

    public static Tensor Quantized(uint rows, uint cols, uint bits = 4) =>
        new() { Dtype = Cq, Shape = new[] { rows, cols }, Bits = bits, Group = 128, Data = new byte[rows * cols * bits / 8 + rows * (cols / 128) * 2] };

    public static Tensor Permutation(int length)
    {
        var tensor = Dense((uint)length);
        for (var i = 0; i < length; i++) BinaryPrimitives.WriteSingleLittleEndian(tensor.Data.AsSpan(i * 4), i);
        return tensor;
    }

    public static Tensor Blob(byte[] bytes) => new() { Dtype = Raw, Shape = new[] { (uint)bytes.Length }, Data = bytes };

    public static byte[] TokenizerBlob(params (byte Type, string Text)[] pieces)
    {
        var stream = new MemoryStream();
        var writer = new BinaryWriter(stream);
        writer.Write((uint)pieces.Length);
        writer.Write(0u);
        writer.Write(1u);
        writer.Write(2u);
        writer.Write(3u);
        writer.Write((byte)0);
        writer.Write((byte)0);
        writer.Write((ushort)0);
        foreach (var (type, text) in pieces)
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            writer.Write(0f);
            writer.Write(type);
            writer.Write((ushort)bytes.Length);
            writer.Write(bytes);
        }
        writer.Flush();
        return stream.ToArray();
    }

    public static byte[] DefaultTokenizer() => TokenizerBlob((0, "<pad>"), (0, "a"), (0, "b"), (0, "c"));

    public SyntheticCact WithMinimalWeights(byte[]? tokenizer = null)
    {
        var dim = Header[7];
        Tensors.Add(Quantized(Header[5], dim));
        for (var l = 0; l < Header[10]; l++) AddLayer(dim);
        for (var i = 0; i < 6; i++) Tensors.Add(Dense(4));
        for (var i = 0; i < 3; i++) Tensors.Add(Quantized(4, dim));
        Tensors.Add(Permutation((int)Header[14]));
        Tensors.Add(Permutation((int)Header[14]));
        Tensors.Add(Dense(dim));
        Tensors.Add(Blob(tokenizer ?? DefaultTokenizer()));
        return this;
    }

    void AddLayer(uint dim)
    {
        Tensors.Add(Dense(dim));
        for (var i = 0; i < 3; i++) Tensors.Add(Quantized(dim, dim));
        for (var i = 0; i < 5; i++) Tensors.Add(Dense(4));
        for (var i = 0; i < 2; i++) Tensors.Add(Quantized(dim, dim));
        Tensors.Add(Dense(dim));
        Tensors.Add(Dense(1));
        Tensors.Add(Dense(4));
        for (var i = 0; i < 13; i++) Tensors.Add(Dense(4));
    }

    public byte[] Build()
    {
        Header[1] = (uint)Tensors.Count;
        Header[2] = (uint)CodebookEntries;
        var tableStart = ModelInfo.HeaderBytes + CodebookEntries * 4;
        var dataStart = tableStart + Tensors.Count * RecordBytes;
        var length = dataStart;
        foreach (var t in Tensors) length += t.Data.Length;
        var bytes = new byte[length];
        MemoryMarshal.AsBytes(Header.AsSpan()).CopyTo(bytes);
        for (var i = 0; i < CodebookEntries; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(ModelInfo.HeaderBytes + i * 4), i < 12 ? (i % 4) - 1.5f : (i - 12) - 7.5f);
        var cursor = dataStart;
        for (var i = 0; i < Tensors.Count; i++)
        {
            var t = Tensors[i];
            var record = bytes.AsSpan(tableStart + i * RecordBytes, RecordBytes);
            record[0] = t.Dtype;
            record[1] = (byte)t.Shape.Length;
            for (var d = 0; d < t.Shape.Length && d < 4; d++) BinaryPrimitives.WriteUInt32LittleEndian(record.Slice(4 + d * 4), t.Shape[d]);
            BinaryPrimitives.WriteUInt64LittleEndian(record.Slice(20), (ulong)(t.OffsetOverride ?? cursor));
            BinaryPrimitives.WriteUInt64LittleEndian(record.Slice(28), (ulong)(t.BytesOverride ?? t.Data.Length));
            BinaryPrimitives.WriteUInt32LittleEndian(record.Slice(36), t.Group);
            BinaryPrimitives.WriteUInt32LittleEndian(record.Slice(40), t.Bits);
            t.Data.CopyTo(bytes, cursor);
            cursor += t.Data.Length;
        }
        return bytes;
    }
}

internal static unsafe class GuardedMemory
{
    const int ProtNone = 0, ProtRead = 1, ProtWrite = 2, MapPrivate = 2, MapAnonymous = 0x20;

    [DllImport("libc", SetLastError = true)]
    static extern nint mmap(nint address, nuint length, int protection, int flags, int fd, nint offset);

    [DllImport("libc", SetLastError = true)]
    static extern int mprotect(nint address, nuint length, int protection);

    public static byte* AgainstGuardPage(byte[] bytes)
    {
        const int page = 4096;
        var pages = (bytes.Length + page - 1) / page + 1;
        var region = mmap(0, (nuint)(pages * page), ProtRead | ProtWrite, MapPrivate | MapAnonymous, -1, 0);
        if (region == -1) throw new InvalidOperationException("mmap failed");
        if (mprotect(region + (pages - 1) * page, page, ProtNone) != 0) throw new InvalidOperationException("mprotect failed");
        var start = (byte*)region + (pages - 1) * page - bytes.Length;
        Marshal.Copy(bytes, 0, (nint)start, bytes.Length);
        return start;
    }
}

internal static unsafe class MemoryProbe
{
    public const string Variable = "NEEDLE_MEMORY_BUGHUNT_PROBE";

#pragma warning disable CA2255
    [ModuleInitializer]
    internal static void Run()
#pragma warning restore CA2255
    {
        var request = Environment.GetEnvironmentVariable(Variable);
        if (request == null) return;
        var split = request.IndexOf(':');
        var mode = request[..split];
        var path = request[(split + 1)..];
        string outcome;
        try
        {
            if (mode == "channel")
            {
                var channel = NeedleChannel.Allocate(int.Parse(path), 1);
                channel.Free();
                throw new InvalidOperationException("allocated");
            }
            var bytes = File.ReadAllBytes(path);
            Model model;
            if (mode.StartsWith("guarded", StringComparison.Ordinal)) model = Model.FromMemory(GuardedMemory.AgainstGuardPage(bytes), (nuint)bytes.Length);
            else model = Model.FromBytes(bytes);
            using (model)
            {
                if (mode.EndsWith("tokenizer", StringComparison.Ordinal))
                {
                    using var tokenizer = new Tokenizer(model);
                }
                if (mode == "vocabulary") model.GetVocabulary();
            }
            outcome = "loaded";
        }
        catch (Exception e)
        {
            outcome = $"threw {e.GetType().Name}: {e.Message.Replace('\n', ' ')}";
        }
        Console.Out.WriteLine($"probe {outcome}");
        Console.Out.Flush();
        Environment.Exit(0);
    }

    public static (int ExitCode, string Outcome, string Errors) Load(string mode, byte[] cact, string? argument = null)
    {
        var file = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".cact");
        File.WriteAllBytes(file, cact);
        try
        {
            var host = Environment.ProcessPath is { } p && Path.GetFileNameWithoutExtension(p) == "dotnet" ? p : Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") ?? "dotnet";
            var assembly = typeof(MemoryProbe).Assembly.Location;
            var stem = Path.Combine(Path.GetDirectoryName(assembly)!, Path.GetFileNameWithoutExtension(assembly));
            var start = new ProcessStartInfo(host) { RedirectStandardOutput = true, RedirectStandardError = true };
            foreach (var part in new[] { "exec", "--runtimeconfig", stem + ".runtimeconfig.json", "--depsfile", stem + ".deps.json", assembly }) start.ArgumentList.Add(part);
            start.Environment[Variable] = $"{mode}:{argument ?? file}";
            start.Environment["DOTNET_DbgEnableMiniDump"] = "0";
            using var process = Process.Start(start)!;
            var errors = process.StandardError.ReadToEndAsync();
            var output = process.StandardOutput.ReadToEndAsync();
            if (!process.WaitForExit(20_000))
            {
                try { process.Kill(entireProcessTree: true); } catch { }
                return (-1, "hung", "");
            }
            var line = Array.FindLast(output.Result.Split('\n'), l => l.StartsWith("probe ", StringComparison.Ordinal))?.Trim() ?? "no probe line";
            var tail = errors.Result.Length > 400 ? errors.Result[..400] : errors.Result;
            return (process.ExitCode, line, tail);
        }
        finally
        {
            File.Delete(file);
        }
    }

    public static void AssertOutcome(string mode, byte[] cact, string prefix, string expected, string? argument = null)
    {
        var (exitCode, outcome, errors) = Load(mode, cact, argument);
        Assert.True(exitCode == 0 && outcome.StartsWith(prefix, StringComparison.Ordinal) && outcome.Contains(expected, StringComparison.Ordinal),
            $"expected \"{prefix}\" naming \"{expected}\"; child exit {exitCode}, {outcome}; stderr: {errors}");
    }

    public static void AssertRejected(string mode, byte[] cact, string expected, string exception = "InvalidDataException") =>
        AssertOutcome(mode, cact, $"probe threw {exception}", expected);
}

public sealed unsafe class MemoryBughuntTests
{
    [Fact]
    public void SyntheticMinimalModelLoads()
    {
        using var model = Model.FromBytes(new SyntheticCact().WithMinimalWeights().Build());
        Assert.Equal(0, model.LayerCount);
        using var tokenizer = new Tokenizer(model);
        Assert.Equal(4, tokenizer.PieceCount);
    }

    [Fact]
    public void SyntheticOneLayerModelLoads()
    {
        using var model = Model.FromBytes(new SyntheticCact(layers: 1).WithMinimalWeights().Build());
        Assert.Equal(1, model.LayerCount);
    }

    static SyntheticCact AllDataInHeader(SyntheticCact cact)
    {
        foreach (var tensor in cact.Tensors)
        {
            tensor.Data = Array.Empty<byte>();
            tensor.OffsetOverride = ModelInfo.HeaderBytes;
            tensor.BytesOverride = 0;
        }
        return cact;
    }

    [Fact]
    public void HugeLayerCountIsRejectedInsteadOfSmashingTheHeap()
    {
        var cact = new SyntheticCact().WithMinimalWeights();
        cact.Header[10] = 0xFFFFFFFF;
        MemoryProbe.AssertRejected("load", cact.Build(), "layers");
    }

    [Fact]
    public void ShortCodebookIsRejectedInsteadOfReadingPastTheFile()
    {
        var cact = new SyntheticCact();
        cact.Tensors.Add(SyntheticCact.Blob(Array.Empty<byte>()));
        cact.CodebookEntries = 0;
        MemoryProbe.AssertRejected("guarded", cact.Build(), "codebook");
    }

    [Fact]
    public void ArchiveWithoutATokenizerTailIsRejectedBeforeWalkingPastTheTensorTable()
    {
        var cact = new SyntheticCact().WithMinimalWeights();
        cact.Tensors.RemoveAt(cact.Tensors.Count - 1);
        MemoryProbe.AssertRejected("guarded", AllDataInHeader(cact).Build(), "tokenizer blob missing", "NotSupportedException");
    }

    [Fact]
    public void DenseTensorLargerThanTheFileIsRejected()
    {
        var cact = new SyntheticCact().WithMinimalWeights();
        var finalNorm = cact.Tensors[^2];
        finalNorm.Shape = new[] { 1u << 17 };
        finalNorm.Data = Array.Empty<byte>();
        MemoryProbe.AssertRejected("guarded", cact.Build(), "tensor 12");
    }

    [Fact]
    public void QuantizedTensorLargerThanTheFileIsRejected()
    {
        var cact = new SyntheticCact().WithMinimalWeights();
        cact.Tensors[0].Shape = new[] { 1u << 20, 128u };
        var error = Assert.Throws<InvalidDataException>(() => Model.FromBytes(cact.Build()));
        Assert.Contains("tensor 0", error.Message);
    }

    [Fact]
    public void TokenizerBlobThatClaimsMorePiecesThanItHoldsIsRejected()
    {
        var blob = SyntheticCact.DefaultTokenizer();
        BinaryPrimitives.WriteUInt32LittleEndian(blob, 400);
        var cact = new SyntheticCact().WithMinimalWeights(blob);
        MemoryProbe.AssertRejected("guarded-tokenizer", cact.Build(), "tokenizer");
    }

    [Fact]
    public void TokenizerByteSymbolThatIsNotHexIsRejectedInsteadOfWritingOutsideTheByteTable()
    {
        var blob = SyntheticCact.TokenizerBlob((0, "<pad>"), (0, "a"), (4, "<0xzz>"), (0, "c"));
        using var model = Model.FromBytes(new SyntheticCact().WithMinimalWeights(blob).Build());
        var error = Assert.Throws<InvalidDataException>(() => new Tokenizer(model).Dispose());
        Assert.Contains("piece 2", error.Message);
    }

    [Fact]
    public void TokenizerPieceLongerThanTheTrieScratchDoesNotSmashTheStack()
    {
        var blob = SyntheticCact.TokenizerBlob((0, "<pad>"), (0, "a"), (0, new string('x', 1000)), (0, "c"));
        MemoryProbe.AssertOutcome("vocabulary", new SyntheticCact().WithMinimalWeights(blob).Build(), "probe loaded", "");
    }

    [Fact]
    public void ChannelCapacityThatOverflowsTheLayoutThrowsInsteadOfWritingPastTheBlock()
    {
        MemoryProbe.AssertOutcome("channel", Array.Empty<byte>(), "probe threw ArgumentOutOfRangeException", "", (1 << 23).ToString());
    }

    [Fact]
    public void ChannelCapacityAboveTheLargestPowerOfTwoThrowsInsteadOfHanging()
    {
        MemoryProbe.AssertOutcome("channel", Array.Empty<byte>(), "probe threw ArgumentOutOfRangeException", "", int.MaxValue.ToString());
    }

    [Fact]
    public void CancellingATicketDoesNotCancelTheTicketThatLaterReusesItsBit()
    {
        var channel = NeedleChannel.Allocate(4, 4);
        try
        {
            Assert.True(channel.TrySubmitQuery("first"u8, out var cancelled));
            channel.Cancel(cancelled);
            Assert.True(channel.TryClaim(out _, out _));
            ulong later = 0;
            for (var i = 0; i < 64; i++)
            {
                Assert.True(channel.TrySubmitQuery("next"u8, out later));
                Assert.True(channel.TryClaim(out _, out _));
            }
            Assert.Equal(cancelled + 64, later);
            Assert.False(channel.IsCancelled(later), $"ticket {later} was never cancelled, but reads as cancelled because ticket {cancelled} was");
        }
        finally
        {
            channel.Free();
        }
    }

    [Fact]
    public void OpeningASharedChannelFileShorterThanItsDeclaredLayoutThrows()
    {
        var path = SharedChannel.NewPath();
        try
        {
            using (SharedChannel.Create(path, 64, 64)) { }
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.ReadWrite)) stream.SetLength(4096);
            Assert.Throws<InvalidDataException>(() => SharedChannel.Open(path).Dispose());
        }
        finally
        {
            File.Delete(path);
        }
    }

    public unsafe struct OversizedCommand
    {
        public fixed byte Payload[2048];
    }

    public struct OversizedSink : Needle.Tools.ICallSink<OversizedCommand>
    {
        public int CallCount => 0;
        public bool TryTakeCall(out OversizedCommand command) { command = default; return false; }
        public void ToolSelected(int tool) { }
        public void PropertySelected(int property) { }
        public void StringByte(byte value) { }
        public void NumberByte(byte value) { }
        public void EnumSelected(int literal) { }
        public void BoolSelected(bool value) { }
        public void NullSelected() { }
        public void ValueCompleted() { }
        public void CallCompleted() { }
    }

    [Fact]
    public void TypedHostRejectsACommandLargerThanTheChannelCallsBuffer()
    {
        using var host = new NeedleHost();
        var options = new HostOptions { ModelPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".cact"), ToolsJson = "[]" };
        var error = Assert.Throws<ArgumentException>(() => host.StartTyped<OversizedSink, OversizedCommand>(options));
        Assert.Contains("2048 bytes", error.Message);
    }
}
