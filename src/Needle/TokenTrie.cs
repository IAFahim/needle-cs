using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Needle;

internal sealed unsafe class TokenTrie : IDisposable
{
    readonly int vocab;
    readonly int maskWords;
    int* edgeStart;
    int* edgeCount;
    byte* edgeByte;
    int* edgeNode;
    int* tokenHead;
    int* tokenNext;
    uint* safeMask;
    int* special;
    int specialCount;
    readonly byte** specialBytes;
    readonly int* specialLength;
    byte* pool;

    public int Words => maskWords;

    public TokenTrie(Tokenizer tokenizer, int vocabSize)
    {
        vocab = vocabSize;
        maskWords = (vocab + 31) / 32;
        var children = new List<SortedDictionary<byte, int>> { new() };
        var heads = new List<int> { -1 };
        var next = new int[vocab];
        var safe = new uint[maskWords];
        var specials = new List<int>();
        var scratch = stackalloc byte[256];
        var poolBytes = new List<byte>();
        var specialOffsets = new List<int>();
        var specialLengths = new List<int>();
        for (var id = 0; id < Math.Min(vocab, tokenizer.PieceCount); id++)
        {
            var length = tokenizer.ConstraintBytes(id, scratch);
            if (length == 0) continue;
            var node = 0;
            var plain = true;
            var quoted = false;
            for (var i = 0; i < length; i++)
            {
                var b = scratch[i];
                if (b < 0x20 || b == 0x7F) plain = false;
                if (b == '"' || b == '\\') { plain = false; quoted = true; }
                if (!children[node].TryGetValue(b, out var child))
                {
                    child = children.Count;
                    children.Add(new SortedDictionary<byte, int>());
                    heads.Add(-1);
                    children[node][b] = child;
                }
                node = child;
            }
            next[id] = heads[node];
            heads[node] = id;
            if (plain) safe[id / 32] |= 1u << (id % 32);
            else if (quoted)
            {
                specials.Add(id);
                specialOffsets.Add(poolBytes.Count);
                specialLengths.Add(length);
                for (var i = 0; i < length; i++) poolBytes.Add(scratch[i]);
            }
        }

        var edges = 0;
        foreach (var c in children) edges += c.Count;
        edgeStart = (int*)Mem.Alloc((nuint)children.Count * 4);
        edgeCount = (int*)Mem.Alloc((nuint)children.Count * 4);
        tokenHead = (int*)Mem.Alloc((nuint)children.Count * 4);
        edgeByte = (byte*)Mem.Alloc((nuint)Math.Max(1, edges));
        edgeNode = (int*)Mem.Alloc((nuint)Math.Max(1, edges) * 4);
        tokenNext = (int*)Mem.Alloc((nuint)vocab * 4);
        safeMask = (uint*)Mem.AllocZeroed((nuint)maskWords * 4);
        var cursor = 0;
        for (var n = 0; n < children.Count; n++)
        {
            edgeStart[n] = cursor;
            edgeCount[n] = children[n].Count;
            tokenHead[n] = heads[n];
            foreach (var pair in children[n]) { edgeByte[cursor] = pair.Key; edgeNode[cursor] = pair.Value; cursor++; }
        }
        for (var i = 0; i < vocab; i++) tokenNext[i] = i < next.Length ? next[i] : -1;
        for (var i = 0; i < maskWords; i++) safeMask[i] = safe[i];
        specialCount = specials.Count;
        special = (int*)Mem.Alloc((nuint)Math.Max(1, specialCount) * 4);
        specialLength = (int*)Mem.Alloc((nuint)Math.Max(1, specialCount) * 4);
        specialBytes = (byte**)Mem.Alloc((nuint)Math.Max(1, specialCount) * (nuint)sizeof(byte*));
        pool = (byte*)Mem.Alloc((nuint)Math.Max(1, poolBytes.Count));
        for (var i = 0; i < poolBytes.Count; i++) pool[i] = poolBytes[i];
        for (var i = 0; i < specialCount; i++)
        {
            special[i] = specials[i];
            specialLength[i] = specialLengths[i];
            specialBytes[i] = pool + specialOffsets[i];
        }
    }

    public void Allow(GrammarProgram* g, in GrammarState state, uint* mask)
    {
        Mem.Clear(mask, (nuint)maskWords * 4);
        if (state.Depth == 0) return;
        fixed (GrammarState* start = &state)
        {
            var top = start->Frames + start->Depth - 1;
            var node = g->Nodes + top->Node * Kind.Fields;
            if (node[0] == Kind.Str && top->Phase == 1 && node[1] < 0)
            {
                for (var i = 0; i < maskWords; i++) mask[i] = safeMask[i];
                for (var i = 0; i < specialCount; i++)
                {
                    var copy = *start;
                    var ok = true;
                    for (var b = 0; b < specialLength[i] && ok; b++) ok = Grammar.Advance(g, &copy, specialBytes[i][b]);
                    if (ok) mask[special[i] / 32] |= 1u << (special[i] % 32);
                }
                return;
            }
            Walk(g, start, mask);
        }
    }

    void Walk(GrammarProgram* g, GrammarState* start, uint* mask)
    {
        const int MaxTokenBytes = 64;
        var states = (GrammarState*)Mem.Alloc((nuint)(MaxTokenBytes + 1) * (nuint)sizeof(GrammarState));
        var nodes = stackalloc int[MaxTokenBytes + 1];
        var cursors = stackalloc int[MaxTokenBytes + 1];
        states[0] = *start;
        nodes[0] = 0;
        cursors[0] = 0;
        var depth = 0;
        while (depth >= 0)
        {
            var node = nodes[depth];
            var index = cursors[depth];
            if (index >= edgeCount[node] || depth + 1 >= MaxTokenBytes) { depth--; continue; }
            cursors[depth]++;
            var edge = edgeStart[node] + index;
            states[depth + 1] = states[depth];
            if (!Grammar.Advance(g, &states[depth + 1], edgeByte[edge])) continue;
            var child = edgeNode[edge];
            for (var t = tokenHead[child]; t >= 0; t = tokenNext[t]) mask[t / 32] |= 1u << (t % 32);
            if (edgeCount[child] == 0) continue;
            depth++;
            nodes[depth] = child;
            cursors[depth] = 0;
        }
        Mem.Free(states);
    }

    public void Dispose()
    {
        Mem.Free(edgeStart);
        Mem.Free(edgeCount);
        Mem.Free(tokenHead);
        Mem.Free(edgeByte);
        Mem.Free(edgeNode);
        Mem.Free(tokenNext);
        Mem.Free(safeMask);
        Mem.Free(special);
        Mem.Free(specialLength);
        Mem.Free(specialBytes);
        Mem.Free(pool);
    }
}
