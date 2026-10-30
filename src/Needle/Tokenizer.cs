using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Needle;

public sealed unsafe class Tokenizer : IDisposable
{
    const byte UserDefined = 3;
    const byte ByteType = 4;
    const byte ControlType = 2;
    const byte UnknownType = 1;
    const int HeaderBytes = 24;
    const int RecordBytes = 7;

    struct Merge
    {
        public float Score;
        public int Left, Right, RightEnd;
    }

    readonly int pieceCount;
    readonly bool addDummyPrefix;
    readonly bool byteFallback;
    readonly int unknownId;
    readonly int maxInput;
    public int EndId { get; }
    public int BeginId { get; }
    public int PadId { get; }

    readonly byte* blob;
    readonly int* pieceOffset;
    readonly ushort* pieceLength;
    readonly float* pieceScore;
    readonly byte* pieceType;
    readonly int* table;
    readonly uint tableMask;
    readonly int* byteIds;
    readonly int* markers;
    readonly int markerCount;

    Arena arena;
    readonly byte* escaped;
    readonly int* symbolEnd, previousSymbol, nextSymbol;
    readonly Merge* heap;

    internal byte* SharedBlob => blob;
    internal int SharedPieceCount => pieceCount;
    internal bool SharedAddDummyPrefix => addDummyPrefix;
    internal bool SharedByteFallback => byteFallback;
    internal int SharedUnknownId => unknownId;
    internal int* SharedPieceOffset => pieceOffset;
    internal ushort* SharedPieceLength => pieceLength;
    internal float* SharedPieceScore => pieceScore;
    internal byte* SharedPieceType => pieceType;
    internal int* SharedTable => table;
    internal uint SharedTableMask => tableMask;
    internal int* SharedByteIds => byteIds;
    internal int* SharedMarkers => markers;
    internal int SharedMarkerCount => markerCount;

    internal Tokenizer(Vocabulary vocabulary, int maxInputBytes)
    {
        var shared = vocabulary.Tables;
        maxInput = maxInputBytes;
        blob = shared.SharedBlob;
        pieceCount = shared.SharedPieceCount;
        PadId = shared.PadId;
        EndId = shared.EndId;
        BeginId = shared.BeginId;
        unknownId = shared.SharedUnknownId;
        addDummyPrefix = shared.SharedAddDummyPrefix;
        byteFallback = shared.SharedByteFallback;
        pieceOffset = shared.SharedPieceOffset;
        pieceLength = shared.SharedPieceLength;
        pieceScore = shared.SharedPieceScore;
        pieceType = shared.SharedPieceType;
        table = shared.SharedTable;
        tableMask = shared.SharedTableMask;
        byteIds = shared.SharedByteIds;
        markers = shared.SharedMarkers;
        markerCount = shared.SharedMarkerCount;

        var worst = (nuint)maxInput * 3 + 8;
        arena = new Arena(worst * 13 + worst * 3 * (nuint)sizeof(Merge) + 4096);
        escaped = arena.TakeUncleared<byte>(worst);
        symbolEnd = arena.TakeUncleared<int>(worst);
        previousSymbol = arena.TakeUncleared<int>(worst);
        nextSymbol = arena.TakeUncleared<int>(worst);
        heap = arena.TakeUncleared<Merge>(worst * 3);
    }

    public Tokenizer(Model model, int maxInputBytes = 1 << 16)
    {
        maxInput = maxInputBytes;
        blob = model.TokenizerBlob;
        var header = (uint*)blob;
        pieceCount = (int)header[0];
        PadId = (int)header[1];
        EndId = (int)header[2];
        BeginId = (int)header[3];
        unknownId = (int)header[4];
        addDummyPrefix = blob[20] != 0;
        byteFallback = blob[21] != 0;

        arena = new Arena((nuint)pieceCount * 64 + (nuint)maxInput * 256 + (4u << 20));
        pieceOffset = arena.Take<int>((nuint)pieceCount);
        pieceLength = arena.Take<ushort>((nuint)pieceCount);
        pieceScore = arena.Take<float>((nuint)pieceCount);
        pieceType = arena.Take<byte>((nuint)pieceCount);
        byteIds = arena.Take<int>(256);
        for (var i = 0; i < 256; i++) byteIds[i] = -1;
        var capacity = 1u;
        while (capacity < (uint)pieceCount * 2) capacity <<= 1;
        table = arena.Take<int>(capacity);
        tableMask = capacity - 1;
        var markerList = arena.Take<int>((nuint)pieceCount);
        var cursor = blob + HeaderBytes;
        for (var id = 0; id < pieceCount; id++)
        {
            pieceScore[id] = *(float*)cursor;
            pieceType[id] = cursor[4];
            pieceLength[id] = *(ushort*)(cursor + 5);
            pieceOffset[id] = (int)(cursor - blob) + RecordBytes;
            cursor += RecordBytes + pieceLength[id];
            Insert(id);
            if (pieceType[id] == ByteType) byteIds[HexValue(Piece(id)[3]) * 16 + HexValue(Piece(id)[4])] = id;
            if (pieceType[id] == UserDefined) markerList[markerCount++] = id;
        }
        markers = markerList;
        SortMarkersByLength();

        var worst = (nuint)maxInput * 3 + 8;
        escaped = arena.TakeUncleared<byte>(worst);
        symbolEnd = arena.TakeUncleared<int>(worst);
        previousSymbol = arena.TakeUncleared<int>(worst);
        nextSymbol = arena.TakeUncleared<int>(worst);
        heap = arena.TakeUncleared<Merge>(worst * 3);
    }

    byte* Piece(int id) => blob + pieceOffset[id];

    static int HexValue(byte c) => c <= (byte)'9' ? c - '0' : (c | 0x20) - 'a' + 10;

    void SortMarkersByLength()
    {
        for (var i = 1; i < markerCount; i++)
        {
            var id = markers[i];
            var j = i - 1;
            while (j >= 0 && pieceLength[markers[j]] < pieceLength[id]) { markers[j + 1] = markers[j]; j--; }
            markers[j + 1] = id;
        }
    }

    static uint Hash(byte* bytes, int length)
    {
        var h = 2166136261u;
        for (var i = 0; i < length; i++) h = (h ^ bytes[i]) * 16777619u;
        return h;
    }

    void Insert(int id)
    {
        var slot = Hash(Piece(id), pieceLength[id]) & tableMask;
        while (table[slot] != 0) slot = (slot + 1) & tableMask;
        table[slot] = id + 1;
    }

    int Lookup(byte* bytes, int length)
    {
        var slot = Hash(bytes, length) & tableMask;
        for (int entry; (entry = table[slot]) != 0; slot = (slot + 1) & tableMask)
        {
            var id = entry - 1;
            if (pieceLength[id] == length && new ReadOnlySpan<byte>(Piece(id), length).SequenceEqual(new ReadOnlySpan<byte>(bytes, length))) return id;
        }
        return -1;
    }

    public int Encode(ReadOnlySpan<byte> utf8, Span<int> output, bool leadingSpace = true)
    {
        if (utf8.Length == 0) return 0;
        if (utf8.Length > maxInput) throw new ArgumentException("input exceeds tokenizer capacity");
        var length = Escape(utf8, leadingSpace && addDummyPrefix);
        var written = 0;
        var segment = 0;
        for (var i = 0; i < length;)
        {
            var marker = escaped[i] == (byte)'<' ? MatchMarker(i, length) : -1;
            if (marker < 0) { i++; continue; }
            written += EncodeSegment(segment, i, output[written..]);
            output[written++] = marker;
            i += pieceLength[marker];
            segment = i;
        }
        return written + EncodeSegment(segment, length, output[written..]);
    }

    int Escape(ReadOnlySpan<byte> utf8, bool dummyPrefix)
    {
        var n = 0;
        if (dummyPrefix) n += WriteSpaceMark(n);
        foreach (var b in utf8)
        {
            if (b == (byte)' ') n += WriteSpaceMark(n);
            else escaped[n++] = b;
        }
        return n;
    }

    int WriteSpaceMark(int at)
    {
        escaped[at] = 0xE2;
        escaped[at + 1] = 0x96;
        escaped[at + 2] = 0x81;
        return 3;
    }

    int MatchMarker(int at, int length)
    {
        for (var m = 0; m < markerCount; m++)
        {
            var id = markers[m];
            var n = pieceLength[id];
            if (at + n <= length && new ReadOnlySpan<byte>(escaped + at, n).SequenceEqual(new ReadOnlySpan<byte>(Piece(id), n))) return id;
        }
        return -1;
    }

    static int CodePointLength(byte lead) => lead < 0x80 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;

    int EncodeSegment(int from, int to, Span<int> output)
    {
        if (from >= to) return 0;
        var heapSize = 0;
        var last = -1;
        for (var i = from; i < to;)
        {
            var end = Math.Min(to, i + CodePointLength(escaped[i]));
            symbolEnd[i] = end;
            previousSymbol[i] = last;
            nextSymbol[i] = -1;
            if (last >= 0) nextSymbol[last] = i;
            last = i;
            i = end;
        }
        for (var i = from; i >= 0; i = nextSymbol[i]) Push(ref heapSize, i);

        while (heapSize > 0)
        {
            var best = Pop(ref heapSize);
            if (symbolEnd[best.Left] == 0 || nextSymbol[best.Left] != best.Right || symbolEnd[best.Right] != best.RightEnd) continue;
            var right = best.Right;
            symbolEnd[best.Left] = symbolEnd[right];
            symbolEnd[right] = 0;
            nextSymbol[best.Left] = nextSymbol[right];
            if (nextSymbol[right] >= 0) previousSymbol[nextSymbol[right]] = best.Left;
            if (previousSymbol[best.Left] >= 0) Push(ref heapSize, previousSymbol[best.Left]);
            Push(ref heapSize, best.Left);
        }

        var count = 0;
        for (var i = from; i >= 0; i = nextSymbol[i])
        {
            var span = symbolEnd[i] - i;
            var id = Lookup(escaped + i, span);
            if (id >= 0) output[count++] = id;
            else if (byteFallback) for (var b = 0; b < span; b++) output[count++] = byteIds[escaped[i + b]];
            else output[count++] = unknownId;
        }
        return count;
    }

    void Push(ref int size, int left)
    {
        var right = nextSymbol[left];
        if (right < 0) return;
        var id = Lookup(escaped + left, symbolEnd[right] - left);
        if (id < 0) return;
        var i = size++;
        var entry = new Merge { Score = pieceScore[id], Left = left, Right = right, RightEnd = symbolEnd[right] };
        while (i > 0)
        {
            var parent = (i - 1) / 2;
            if (!Before(entry, heap[parent])) break;
            heap[i] = heap[parent];
            i = parent;
        }
        heap[i] = entry;
    }

    Merge Pop(ref int size)
    {
        var top = heap[0];
        var tail = heap[--size];
        var i = 0;
        for (;;)
        {
            var child = 2 * i + 1;
            if (child >= size) break;
            if (child + 1 < size && Before(heap[child + 1], heap[child])) child++;
            if (!Before(heap[child], tail)) break;
            heap[i] = heap[child];
            i = child;
        }
        if (size > 0) heap[i] = tail;
        return top;
    }

    static bool Before(in Merge a, in Merge b) => a.Score > b.Score || (a.Score == b.Score && a.Left < b.Left);

    public int Append(int id, Span<byte> output, bool first)
    {
        if (pieceType[id] == ControlType || pieceType[id] == UnknownType) return 0;
        if (pieceType[id] == ByteType)
        {
            output[0] = (byte)(HexValue(Piece(id)[3]) * 16 + HexValue(Piece(id)[4]));
            return 1;
        }
        var n = 0;
        var bytes = Piece(id);
        for (var i = 0; i < pieceLength[id];)
        {
            if (bytes[i] == 0xE2 && i + 2 < pieceLength[id] && bytes[i + 1] == 0x96 && bytes[i + 2] == 0x81)
            {
                if (!(first && addDummyPrefix && i == 0)) output[n++] = (byte)' ';
                i += 3;
            }
            else output[n++] = bytes[i++];
        }
        return n;
    }

    public int PieceCount => pieceCount;

    public string PieceText(int id) => Encoding.UTF8.GetString(Piece(id), pieceLength[id]);

    public bool IsPlain(int id) => pieceType[id] == 0 || pieceType[id] == ByteType;

    public bool IsConstrainable(int id) => pieceType[id] == 0 || pieceType[id] == ByteType;

    public int ConstraintBytes(int id, byte* destination)
    {
        if (!IsConstrainable(id)) return 0;
        if (pieceType[id] == ByteType)
        {
            destination[0] = (byte)(HexValue(Piece(id)[3]) * 16 + HexValue(Piece(id)[4]));
            return 1;
        }
        var n = 0;
        var bytes = Piece(id);
        for (var i = 0; i < pieceLength[id];)
        {
            if (bytes[i] == 0xE2 && i + 2 < pieceLength[id] && bytes[i + 1] == 0x96 && bytes[i + 2] == 0x81) { destination[n++] = (byte)' '; i += 3; }
            else destination[n++] = bytes[i++];
        }
        return n;
    }

    public bool IsStop(int id) => id == EndId || (pieceType[id] == UserDefined && Matches(id, "<|im_end|>"u8));

    bool Matches(int id, ReadOnlySpan<byte> text) =>
        new ReadOnlySpan<byte>(Piece(id), pieceLength[id]).SequenceEqual(text);

    public void Dispose() => arena.Free();
}
