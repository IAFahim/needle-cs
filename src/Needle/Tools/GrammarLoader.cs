using System;

namespace Needle;

internal static unsafe class GrammarLoader
{
    public static GrammarProgram* Load(
        ReadOnlySpan<int> nodes,
        ReadOnlySpan<int> properties,
        ReadOnlySpan<int> literals,
        ReadOnlySpan<int> unions,
        ReadOnlySpan<int> tools,
        ReadOnlySpan<byte> bytes,
        int root,
        int toolCount)
    {
        var program = (GrammarProgram*)Mem.AllocZeroed((nuint)sizeof(GrammarProgram));
        program->Nodes = CopyInts(nodes);
        program->Properties = CopyInts(properties);
        program->Literals = CopyInts(literals);
        program->Unions = CopyInts(unions);
        program->Tools = CopyInts(tools);
        program->Bytes = CopyBytes(bytes);
        program->Root = root;
        program->ToolCount = toolCount;
        return program;
    }

    public static void Free(GrammarProgram* program)
    {
        Mem.Free(program->Nodes);
        Mem.Free(program->Properties);
        Mem.Free(program->Literals);
        Mem.Free(program->Unions);
        Mem.Free(program->Tools);
        Mem.Free(program->Bytes);
        Mem.Free(program);
    }

    static int* CopyInts(ReadOnlySpan<int> span)
    {
        var pointer = (int*)Mem.Alloc((nuint)Math.Max(1, span.Length) * sizeof(int));
        span.CopyTo(new Span<int>(pointer, span.Length));
        return pointer;
    }

    static byte* CopyBytes(ReadOnlySpan<byte> span)
    {
        var pointer = (byte*)Mem.Alloc((nuint)Math.Max(1, span.Length));
        span.CopyTo(new Span<byte>(pointer, span.Length));
        return pointer;
    }
}
