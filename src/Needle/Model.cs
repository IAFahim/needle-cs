using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace Needle;

internal readonly unsafe struct Cq
{
    public const int GroupSize = 128;

    public readonly byte* Indices;
    public readonly ushort* Norms;
    public readonly Codebook* Book;
    public readonly int Rows;
    public readonly int Cols;
    public readonly int Bits;

    public Cq(byte* indices, ushort* norms, Codebook* book, int rows, int cols, int bits)
    {
        Indices = indices;
        Norms = norms;
        Book = book;
        Rows = rows;
        Cols = cols;
        Bits = bits;
    }

    public int Groups => Cols / GroupSize;
    public int RowBytes => Cols * Bits / 8;

    public Cq Slice(int first, int count) =>
        new(Indices + (long)first * RowBytes, Norms + (long)first * Groups, Book, count, Cols, Bits);
}

internal unsafe struct Codebook
{
    public const int Bias = 63;

    public fixed float Levels[16];
    public fixed byte Bytes[32];
    public float ByteScale;

    public static void Fill(Codebook* book, float* levels, int count)
    {
        var peak = 0f;
        for (var i = 0; i < count; i++) peak = MathF.Max(peak, MathF.Abs(levels[i]));
        book->ByteScale = peak / Bias;
        for (var i = 0; i < 16; i++) book->Levels[i] = levels[i % count];
        for (var i = 0; i < 32; i++)
            book->Bytes[i] = i % 16 < count ? (byte)(MathF.Round(levels[i % 16] / book->ByteScale) + Bias) : (byte)0;
    }
}

internal static class BitMath
{
    public static int BitLength(this int value)
    {
        var bits = 0;
        for (var v = (uint)value; v != 0; v >>= 1) bits++;
        return bits;
    }
}

internal readonly unsafe struct Geometry
{
    public readonly int Vocab, Dim, Heads, KvHeads, Layers, QkDim, VDim, MaxSeq, HadaN, Lanes;
    public readonly int SlidingWindow, ConvTaps, EngramSlots, EngramSub, EngramTables;
    public readonly int EngramConvTaps, EngramDilation, EngramSeedHeads, KvWindow;
    public readonly int OrderCount, SiteCount;
    public readonly ulong GlobalMask;
    public readonly float RopeTheta;
    public readonly int* Orders;
    public readonly int* Sites;

    public Geometry(uint* h, int* orders, int* sites)
    {
        KvWindow = (int)h[3];
        Vocab = (int)h[5];
        Dim = (int)h[7];
        Heads = (int)h[8];
        KvHeads = (int)h[9];
        Layers = (int)h[10];
        QkDim = (int)h[11];
        VDim = (int)h[12];
        MaxSeq = (int)h[13];
        HadaN = (int)h[14];
        Lanes = (int)h[15];
        SlidingWindow = (int)h[16];
        GlobalMask = h[17] | ((ulong)h[18] << 32);
        ConvTaps = (int)h[19];
        EngramSlots = (int)h[20];
        EngramSub = (int)h[21];
        EngramTables = (int)h[22];
        EngramConvTaps = (int)h[23];
        EngramDilation = (int)h[24];
        EngramSeedHeads = (int)h[25];
        OrderCount = (int)h[26];
        SiteCount = (int)h[31];
        RopeTheta = ((float*)h)[48];
        Orders = orders;
        Sites = sites;
        for (var i = 0; i < OrderCount; i++) orders[i] = (int)h[27 + i];
        for (var i = 0; i < SiteCount; i++) sites[i] = (int)h[32 + i];
    }

    public int HadaRows => 1 << ((HadaN - 1).BitLength() / 2);
    public int HadaCols => HadaN / HadaRows;
    public int OutputWidth => Heads * VDim;
    public int QueryWidth => Heads * QkDim;
    public int KeyWidth => KvHeads * QkDim;
    public int ValueWidth => KvHeads * VDim;
    public int HeadsPerKv => Heads / KvHeads;
    public int EngramHeads => EngramTables / OrderCount;
    public bool IsGlobal(int layer) => (GlobalMask >> layer & 1) != 0;
    public int SiteOf(int layer)
    {
        for (var i = 0; i < SiteCount; i++) if (Sites[i] == layer) return i;
        return -1;
    }
}

internal readonly unsafe struct LayerWeights
{
    public readonly float* NormIn;
    public readonly Cq Q, K, V;
    public readonly float* QTaps, KTaps, VTaps, QNorm, KNorm;
    public readonly Cq Gate, Out;
    public readonly float* PostNorm;
    public readonly float AttnGate;
    public readonly float* PreHada;
    public readonly MlpWeights Mlp;

    public LayerWeights(ref Reader r)
    {
        NormIn = r.Dense();
        Q = r.Quantized();
        K = r.Quantized();
        V = r.Quantized();
        QTaps = r.Dense();
        KTaps = r.Dense();
        VTaps = r.Dense();
        QNorm = r.Dense();
        KNorm = r.Dense();
        Gate = r.Quantized();
        Out = r.Quantized();
        PostNorm = r.Dense();
        AttnGate = 1f / (1f + MathF.Exp(-*r.Dense()));
        PreHada = r.Dense();
        Mlp = new MlpWeights(ref r);
    }
}

internal readonly unsafe struct HyperConnections
{
    public readonly float* APre, APost, ARes, BPre, BPost, BRes;
    public readonly Cq PhiPre, PhiPost, PhiRes;
    public readonly int* PermA, PermB;

    public HyperConnections(ref Reader r, int permLength)
    {
        APre = r.Dense();
        APost = r.Dense();
        ARes = r.Dense();
        BPre = r.Dense();
        BPost = r.Dense();
        BRes = r.Dense();
        PhiPre = r.Quantized();
        PhiPost = r.Quantized();
        PhiRes = r.Quantized();
        PermA = permLength > 0 ? r.Indices(permLength) : null;
        PermB = permLength > 0 ? r.Indices(permLength) : null;
    }
}

internal readonly unsafe struct EngramSite
{
    public readonly Cq Tables, Key, Value;
    public readonly float* Taps;

    public EngramSite(ref Reader r)
    {
        Tables = r.Quantized();
        Key = r.Quantized();
        Value = r.Quantized();
        Taps = r.Dense();
    }
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
internal readonly struct Record
{
    public readonly byte Dtype, Ndim;
    public readonly ushort Pad;
    public readonly uint Shape0, Shape1, Shape2, Shape3;
    public readonly ulong Offset, Bytes;
    public readonly uint Group, Bits;
}

internal unsafe struct Reader
{
    public const byte Fp16 = 1, Fp32 = 2, Cq = 3, Raw = 4;

    readonly byte* map;
    readonly Codebook* book2;
    readonly Codebook* book4;
    Record* next;
    Arena arena;

    public Reader(byte* map, Record* first, Arena arena, float* codebook)
    {
        this.map = map;
        next = first;
        this.arena = arena;
        book2 = this.arena.Take<Codebook>(1);
        book4 = this.arena.Take<Codebook>(1);
        Codebook.Fill(book2, codebook, 4);
        Codebook.Fill(book4, codebook + 12, 16);
    }

    public Arena Arena => arena;

    public float* Rows(Cq tensor)
    {
        var destination = arena.Take<float>((nuint)tensor.Rows * (nuint)tensor.Cols);
        for (var row = 0; row < tensor.Rows; row++) Quant.GatherRow(tensor, row, destination + (long)row * tensor.Cols);
        return destination;
    }

    public Record* Next => next;

    public T* Take<T>(nuint count) where T : unmanaged => arena.Take<T>(count);

    public float* Dense()
    {
        var record = next++;
        var count = Elements(record);
        var destination = arena.Take<float>(count);
        var source = map + record->Offset;
        if (record->Dtype == Fp16) for (nuint i = 0; i < count; i++) destination[i] = Scalar.HalfToSingle(((ushort*)source)[i]);
        else if (record->Dtype == Fp32) Buffer.MemoryCopy(source, destination, count * 4, count * 4);
        else throw new InvalidDataException("dense tensor expected");
        return destination;
    }

    public int* Indices(int expected)
    {
        var record = next++;
        if (record->Dtype != Fp32 || (int)Elements(record) != expected) throw new InvalidDataException("permutation expected");
        var destination = arena.Take<int>((nuint)expected);
        for (var i = 0; i < expected; i++) destination[i] = (int)((float*)(map + record->Offset))[i];
        return destination;
    }

    public Needle.Cq Quantized()
    {
        var record = next++;
        if (record->Dtype != Cq || (record->Bits != 2 && record->Bits != 4)) throw new InvalidDataException("2- or 4-bit CQ tensor expected");
        var rows = (int)record->Shape0;
        var cols = (int)record->Shape1;
        if (cols % Needle.Cq.GroupSize != 0 || record->Group != Needle.Cq.GroupSize) throw new InvalidDataException("CQ group of 128 expected");
        var indices = map + record->Offset;
        var norms = (ushort*)(indices + (long)rows * cols * record->Bits / 8);
        return new Needle.Cq(indices, norms, record->Bits == 2 ? book2 : book4, rows, cols, (int)record->Bits);
    }

    static nuint Elements(Record* record)
    {
        nuint count = 1;
        var shape = &record->Shape0;
        for (var i = 0; i < record->Ndim; i++) count *= shape[i];
        return count;
    }
}

public sealed unsafe class Model : IDisposable
{
    internal readonly Geometry Geometry;
    internal readonly Cq Embedding;
    internal readonly LayerWeights* Blocks;
    internal readonly HyperConnections Hyper;
    internal readonly EngramSite* Sites;
    internal readonly float* FinalNorm;
    internal readonly ProbeHead* Confidence;
    internal readonly ProbeHead* EmbeddingHead;
    internal readonly AudioWeights* Audio;
    internal readonly byte* TokenizerBlob;
    internal readonly nuint TokenizerLength;

    readonly ModelMemory memory;
    Arena arena;
    int disposed;

    Model(ModelMemory memory)
    {
        this.memory = memory;
        var map = memory.Address;
        var length = memory.Length;
        var header = (uint*)map;
        RequireWellFormed(map, length);
        var count = (int)header[1];
        var codebook = (float*)(map + ModelInfo.HeaderBytes);
        var records = (Record*)(map + ModelInfo.HeaderBytes + header[2] * 4);

        nuint denseBytes = 0;
        for (var i = 0; i < count; i++)
            if (records[i].Dtype == Reader.Fp16 || records[i].Dtype == Reader.Fp32) denseBytes += (nuint)records[i].Bytes * 2 + 128;
        arena = new Arena(denseBytes + (1u << 20));

        var orders = arena.Take<int>(4);
        var sites = arena.Take<int>(16);
        Geometry = new Geometry(header, orders, sites);
        Require(Geometry.QkDim == 48 && Geometry.VDim == 64 && Geometry.Lanes == 4,
            $"unsupported head geometry: found QkDim={Geometry.QkDim}, VDim={Geometry.VDim}, Lanes={Geometry.Lanes}; this build supports QkDim=48, VDim=64, Lanes=4");
        Require(Geometry.HadaCols % 8 == 0 && Geometry.HadaRows * Geometry.HadaCols == Geometry.HadaN,
            $"unsupported MLP geometry: found HadaN={Geometry.HadaN}; this build supports sizes that split into rows by columns with columns divisible by 8");
        Require(Geometry.ConvTaps == 3 && Geometry.EngramConvTaps == 4 && Geometry.Dim % Cq.GroupSize == 0,
            $"unsupported convolution geometry: found ConvTaps={Geometry.ConvTaps}, EngramConvTaps={Geometry.EngramConvTaps}, Dim={Geometry.Dim}; this build supports ConvTaps=3, EngramConvTaps=4 and Dim divisible by {Cq.GroupSize}");

        var reader = new Reader(map, records, arena, codebook);
        Embedding = reader.Quantized();
        Blocks = reader.Take<LayerWeights>((nuint)Geometry.Layers);
        for (var l = 0; l < Geometry.Layers; l++) Blocks[l] = new LayerWeights(ref reader);
        Hyper = new HyperConnections(ref reader, Geometry.HadaN);
        Sites = reader.Take<EngramSite>((nuint)Geometry.SiteCount);
        for (var s = 0; s < Geometry.SiteCount; s++) Sites[s] = new EngramSite(ref reader);
        FinalNorm = reader.Dense();
        var tail = records + count - 1;
        if (reader.Next->Dtype == Reader.Fp32 && reader.Next->Ndim == 1 && reader.Next->Shape0 == 13)
        {
            Audio = reader.Take<AudioWeights>(1);
            *Audio = new AudioWeights(ref reader, Geometry.Layers, Geometry.HadaN);
        }
        else if (reader.Next != tail)
        {
            var manifest = reader.Dense();
            for (var h = 0; reader.Next != tail; h++)
            {
                var code = (int)MathF.Round(manifest[h]);
                var head = reader.Take<ProbeHead>(1);
                *head = new ProbeHead(ref reader, code, Geometry.Layers, Geometry.Dim);
                if (code == ProbeHead.Confidence) Confidence = head;
                else if (code == ProbeHead.Embedding) EmbeddingHead = head;
            }
        }

        var tokenizer = records + count - 1;
        Require(tokenizer->Dtype == Reader.Raw, $"tokenizer blob missing: the last tensor has dtype {tokenizer->Dtype}, expected {Reader.Raw}");
        TokenizerBlob = map + tokenizer->Offset;
        TokenizerLength = (nuint)tokenizer->Bytes;
        arena = reader.Arena;
        Info = new ModelInfo(Geometry, header[0], (long)length, count, (int)header[2], (long)tokenizer->Bytes, Audio != null, Confidence != null, EmbeddingHead != null);
    }

    public ModelInfo Info { get; }

    static void RequireWellFormed(byte* map, nuint length)
    {
        if (length < ModelInfo.HeaderBytes)
            throw Malformed($"not a Needle 3 .cact archive: the file has {length} bytes, shorter than the {ModelInfo.HeaderBytes}-byte header");
        var header = (uint*)map;
        if (header[0] != ModelInfo.SupportedMagic)
            throw Malformed($"not a supported Needle 3 .cact archive: found magic 0x{header[0]:X8}, supported magic is 0x{ModelInfo.SupportedMagic:X8}");
        var count = header[1];
        var tableEnd = (ulong)ModelInfo.HeaderBytes + (ulong)header[2] * 4 + (ulong)count * (ulong)sizeof(Record);
        if (count == 0 || tableEnd > length)
            throw Malformed($"truncated or corrupt .cact archive: the header declares {count} tensors and {header[2]} codebook entries, needing {tableEnd} bytes of tables, but the file has {length} bytes");
        var records = (Record*)(map + ModelInfo.HeaderBytes + (nuint)header[2] * 4);
        for (uint i = 0; i < count; i++)
            if (records[i].Offset > length || records[i].Bytes > length - records[i].Offset)
                throw Malformed($"truncated .cact archive: tensor {i} ends at byte {records[i].Offset + records[i].Bytes}, but the file has {length} bytes");
        if (header[26] > 4 || header[31] > 16)
            throw new NotSupportedException($"unsupported engram layout: found {header[26]} orders and {header[31]} sites; this build supports at most 4 orders and 16 sites. {ModelInfo.RefreshGuidance}");
    }

    static InvalidDataException Malformed(string facts) => new($"{facts}. {ModelInfo.RefreshGuidance}");

    public static Model Open(string path) => Open(path, true);

    public static Model Open(string path, bool prefault) => From(ModelMemory.Map(path), prefault);

    public static Model FromMemory(byte* data, nuint length) => From(ModelMemory.Borrow(data, length), false);

    public static Model FromBytes(byte[] bytes) => From(ModelMemory.Copy(bytes), false);

    public static Model FromStream(Stream stream) => From(ModelMemory.Read(stream), false);

    public void Prefault() => memory.Prefault();

    static Model From(ModelMemory memory, bool prefault)
    {
        try
        {
            var model = new Model(memory);
            if (prefault) model.Prefault();
            return model;
        }
        catch
        {
            memory.Dispose();
            throw;
        }
    }

    public int VocabSize => Geometry.Vocab;
    public int LayerCount => Geometry.Layers;

    static void Require(bool condition, string message)
    {
        if (!condition) throw new NotSupportedException($"{message}. {ModelInfo.RefreshGuidance}");
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        memory.Dispose();
        arena.Free();
    }
}
