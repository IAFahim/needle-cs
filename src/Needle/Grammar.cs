using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Needle;

internal static class Kind
{
    public const int Any = 0, Str = 1, Int = 2, Num = 3, Bool = 4, Null = 5, Arr = 6, Obj = 7, Enum = 8, Union = 9, Root = 10, Call = 11;
    public const int AnyNode = 0, StrNode = 1, IntNode = 2, NumNode = 3, BoolNode = 4, NullNode = 5, AnyArrayNode = 6, AnyObjectNode = 7, CallNode = 8;
    public const int Fields = 8;
}

internal unsafe struct GrammarProgram
{
    public int* Nodes;
    public int* Properties;
    public int* Literals;
    public int* Unions;
    public int* Tools;
    public byte* Bytes;
    public int Root;
    public int ToolCount;
}

[StructLayout(LayoutKind.Sequential)]
internal struct Frame
{
    public int Node, Phase, Aux, Aux2;
    public ulong Bits;
}

internal unsafe struct GrammarState
{
    public const int MaxDepth = 16;
    public int Depth;
    public fixed ulong Raw[MaxDepth * 3];

    public Frame* Frames
    {
        get
        {
            fixed (ulong* p = Raw) return (Frame*)p;
        }
    }
}

internal sealed unsafe class Grammar : IDisposable
{
    static readonly byte[] CallHead = Encoding.ASCII.GetBytes("{\"name\":\"");
    static readonly byte[] CallMid = Encoding.ASCII.GetBytes(",\"arguments\":");

    public GrammarProgram* Program;
    readonly List<byte[]> toolNames = new();
    readonly List<int> nodes = new();
    readonly List<int> properties = new();
    readonly List<int> literals = new();
    readonly List<int> unions = new();
    readonly List<int> tools = new();
    readonly List<byte> bytes = new();

    public Grammar(IReadOnlyList<JsonValue> toolSchemas)
    {
        AddNode(Kind.Any, 0, 0, 0, 0);
        AddNode(Kind.Str, -1, 0, 0, 0);
        AddNode(Kind.Int, 0, 0, 0, 0);
        AddNode(Kind.Num, 0, 0, 0, 0);
        AddNode(Kind.Bool, 0, 0, 0, 0);
        AddNode(Kind.Null, 0, 0, 0, 0);
        AddNode(Kind.Arr, Kind.AnyNode, 0, -1, 0);
        AddNode(Kind.Obj, 0, 0, 0, 1);
        AddNode(Kind.Call, 0, 0, 0, 0);
        var root = AddNode(Kind.Root, 0, 0, 0, 0);
        var arguments = new List<int>();
        foreach (var tool in toolSchemas)
        {
            var parameters = tool.Get("parameters") ?? JsonValue.Object(new List<KeyValuePair<string, JsonValue>>());
            arguments.Add(CompileObject(parameters, forceObject: true));
        }
        var first = tools.Count / 4;
        for (var i = 0; i < toolSchemas.Count; i++)
        {
            var name = Encoding.UTF8.GetBytes(toolSchemas[i].GetString("name")!);
            tools.Add(bytes.Count);
            tools.Add(name.Length);
            tools.Add(arguments[i]);
            tools.Add(0);
            bytes.AddRange(name);
        }
        nodes[root * Kind.Fields + 1] = toolSchemas.Count;
        nodes[root * Kind.Fields + 2] = first;

        Program = (GrammarProgram*)Mem.AllocZeroed((nuint)sizeof(GrammarProgram));
        Program->Nodes = Copy(nodes);
        Program->Properties = Copy(properties);
        Program->Literals = Copy(literals);
        Program->Unions = Copy(unions);
        Program->Tools = Copy(tools);
        Program->Bytes = (byte*)Mem.Alloc((nuint)Math.Max(1, bytes.Count));
        for (var i = 0; i < bytes.Count; i++) Program->Bytes[i] = bytes[i];
        Program->Root = root;
        Program->ToolCount = toolSchemas.Count;
        TableLengths = (nodes.Count, properties.Count, literals.Count, unions.Count, tools.Count, bytes.Count);
    }

    internal readonly (int Nodes, int Properties, int Literals, int Unions, int Tools, int Bytes) TableLengths;

    internal Grammar(GrammarProgram* adopted) => Program = adopted;

    static int* Copy(List<int> list)
    {
        var p = (int*)Mem.Alloc((nuint)Math.Max(1, list.Count) * sizeof(int));
        for (var i = 0; i < list.Count; i++) p[i] = list[i];
        return p;
    }

    int AddNode(int kind, int a, int b, int c, int d)
    {
        var index = nodes.Count / Kind.Fields;
        nodes.Add(kind);
        nodes.Add(a);
        nodes.Add(b);
        nodes.Add(c);
        nodes.Add(d);
        nodes.Add(0);
        nodes.Add(0);
        nodes.Add(0);
        return index;
    }

    int Literal(string text)
    {
        var raw = Encoding.UTF8.GetBytes(text);
        var index = literals.Count / 2;
        literals.Add(bytes.Count);
        literals.Add(raw.Length);
        bytes.AddRange(raw);
        return index;
    }

    int CompileObject(JsonValue schema, bool forceObject)
    {
        var props = schema.Get("properties");
        if (props == null || props.Kind != JsonKind.Object || props.Members.Count > 32)
            return forceObject && props == null ? NewObject(0, 0, 0) : Kind.AnyObjectNode;
        var required = 0;
        var names = new List<string>();
        var children = new List<int>();
        foreach (var member in props.Members) names.Add(member.Key);
        if (schema.Get("required") is { Kind: JsonKind.Array } list)
            foreach (var item in list.Items)
            {
                var index = names.IndexOf(item.Text);
                if (index >= 0) required |= 1 << index;
            }
        foreach (var member in props.Members) children.Add(Compile(member.Value));
        var first = properties.Count / 4;
        for (var i = 0; i < names.Count; i++)
        {
            var raw = Encoding.UTF8.GetBytes(names[i]);
            properties.Add(bytes.Count);
            properties.Add(raw.Length);
            properties.Add(children[i]);
            properties.Add(0);
            bytes.AddRange(raw);
        }
        return NewObject(names.Count, first, required);
    }

    int NewObject(int count, int first, int required) => AddNode(Kind.Obj, count, first, required, 0);

    int Compile(JsonValue schema)
    {
        if (schema.Kind != JsonKind.Object) return Kind.AnyNode;
        if (schema.Get("enum") is { Kind: JsonKind.Array } options && options.Items.Count is > 0 and <= 64) return CompileEnum(options.Items);
        if (schema.Get("const") is { } constant) return CompileEnum(new List<JsonValue> { constant });
        var branches = schema.Get("anyOf") ?? schema.Get("oneOf");
        if (branches is { Kind: JsonKind.Array } && branches.Items.Count is > 0 and <= 16) return CompileUnion(branches.Items.ConvertAll(Compile));
        var type = schema.Get("type");
        if (type is { Kind: JsonKind.Array })
        {
            var each = new List<int>();
            foreach (var t in type.Items) each.Add(CompileTyped(schema, t.Text));
            return each.Count == 1 ? each[0] : CompileUnion(each);
        }
        if (type is { Kind: JsonKind.String }) return CompileTyped(schema, type.Text);
        if (schema.Get("properties") != null) return CompileTyped(schema, "object");
        if (schema.Get("items") != null) return CompileTyped(schema, "array");
        return Kind.AnyNode;
    }

    int CompileTyped(JsonValue schema, string type)
    {
        switch (type)
        {
            case "string": return AddNode(Kind.Str, Integer(schema, "maxLength", -1), Integer(schema, "minLength", 0), 0, 0);
            case "integer": return Kind.IntNode;
            case "number": return Kind.NumNode;
            case "boolean": return Kind.BoolNode;
            case "null": return Kind.NullNode;
            case "array":
            {
                var item = schema.Get("items") is { } items ? Compile(items) : Kind.AnyNode;
                return AddNode(Kind.Arr, item, Integer(schema, "minItems", 0), Integer(schema, "maxItems", -1), 0);
            }
            case "object": return CompileObject(schema, forceObject: false);
            default: return Kind.AnyNode;
        }
    }

    static int Integer(JsonValue schema, string key, int fallback) =>
        schema.Get(key) is { Kind: JsonKind.Number } n && int.TryParse(n.Text, out var v) ? v : fallback;

    int CompileEnum(List<JsonValue> values)
    {
        var first = literals.Count / 2;
        foreach (var value in values) Literal(value.ToString());
        return AddNode(Kind.Enum, values.Count, first, 0, 0);
    }

    int CompileUnion(List<int> branches)
    {
        var first = unions.Count;
        unions.AddRange(branches);
        return AddNode(Kind.Union, branches.Count, first, 0, 0);
    }

    public static GrammarState Start(GrammarProgram* g)
    {
        var state = new GrammarState();
        Push(&state, g->Root);
        return state;
    }

    public static bool Complete(in GrammarState state) => state.Depth == 0;

    static bool Push(GrammarState* s, int node)
    {
        if (s->Depth >= GrammarState.MaxDepth) return false;
        var f = s->Frames + s->Depth++;
        *f = new Frame { Node = node };
        return true;
    }

    static bool FirstByteOk(GrammarProgram* g, int node, byte c)
    {
        var n = g->Nodes + node * Kind.Fields;
        switch (n[0])
        {
            case Kind.Any: return true;
            case Kind.Str: return c == '"';
            case Kind.Int:
            case Kind.Num: return c == '-' || (c >= '0' && c <= '9');
            case Kind.Bool: return c == 't' || c == 'f';
            case Kind.Null: return c == 'n';
            case Kind.Arr: return c == '[';
            case Kind.Obj: return c == '{';
            case Kind.Enum:
                for (var i = 0; i < n[1]; i++) if (g->Bytes[g->Literals[(n[2] + i) * 2]] == c) return true;
                return false;
            case Kind.Union:
                for (var i = 0; i < n[1]; i++) if (FirstByteOk(g, g->Unions[n[2] + i], c)) return true;
                return false;
            default: return false;
        }
    }

    const int Reject = 0, Consumed = 1, Reprocess = 2;

    public static bool Advance(GrammarProgram* g, GrammarState* s, byte c)
    {
        var sink = new Tools.NoSink();
        return Advance(g, s, c, ref sink);
    }

    public static bool Advance<TSink>(GrammarProgram* g, GrammarState* s, byte c, ref TSink sink) where TSink : struct, Tools.IGrammarSink
    {
        for (var guard = 0; guard < 12; guard++)
        {
            if (s->Depth == 0) return false;
            var result = Step(g, s, s->Frames + s->Depth - 1, c, ref sink);
            if (result != Reprocess) return result == Consumed;
        }
        return false;
    }

    static int Pop(GrammarState* s, int result)
    {
        s->Depth--;
        return result;
    }

    static int Step<TSink>(GrammarProgram* g, GrammarState* s, Frame* f, byte c, ref TSink sink) where TSink : struct, Tools.IGrammarSink
    {
        var n = g->Nodes + f->Node * Kind.Fields;
        switch (n[0])
        {
            case Kind.Root: return StepRoot(g, s, f, c);
            case Kind.Call: return StepCall(g, s, f, c, ref sink);
            case Kind.Obj: return n[4] != 0 ? StepFreeObject(s, f, c) : StepObject(g, s, f, n, c, ref sink);
            case Kind.Arr: return StepArray(s, f, n, c);
            case Kind.Str: return StepString(f, n, s, c, ref sink);
            case Kind.Int:
            case Kind.Num: return StepNumber(s, f, n[0] == Kind.Num, c, ref sink);
            case Kind.Bool: return StepBool(s, f, c, ref sink);
            case Kind.Null: return StepNull(s, f, c, ref sink);
            case Kind.Enum: return StepEnum(g, s, f, n, c, ref sink);
            case Kind.Union: return StepUnion(g, s, f, n, c);
            case Kind.Any: return StepAny(s, f, c);
            default: return Reject;
        }
    }

    static int StepRoot(GrammarProgram* g, GrammarState* s, Frame* f, byte c)
    {
        switch (f->Phase)
        {
            case 0:
                if (c != '[') return Reject;
                f->Phase = 1;
                return Consumed;
            case 1:
                if (c == ']') return Pop(s, Consumed);
                goto case 3;
            case 3:
                if (c != '{' || g->ToolCount == 0) return Reject;
                f->Phase = 2;
                if (!Push(s, Kind.CallNode)) return Reject;
                return Reprocess;
            default:
                if (c == ',') { f->Phase = 3; return Consumed; }
                return c == ']' ? Pop(s, Consumed) : Reject;
        }
    }

    static int StepCall<TSink>(GrammarProgram* g, GrammarState* s, Frame* f, byte c, ref TSink sink) where TSink : struct, Tools.IGrammarSink
    {
        var root = g->Nodes + g->Root * Kind.Fields;
        switch (f->Phase)
        {
            case 0:
                if (c != CallHead[f->Aux]) return Reject;
                if (++f->Aux == CallHead.Length) { f->Phase = 1; f->Aux = 0; f->Bits = ToolMask(g); }
                return Consumed;
            case 1:
            {
                var candidates = 0UL;
                if (c == '"')
                {
                    for (var t = 0; t < g->ToolCount; t++)
                        if ((f->Bits >> t & 1) != 0 && g->Tools[(root[2] + t) * 4 + 1] == f->Aux) { f->Aux2 = t; f->Phase = 2; f->Aux = 0; sink.ToolSelected(t); return Consumed; }
                    return Reject;
                }
                for (var t = 0; t < g->ToolCount; t++)
                {
                    var tool = g->Tools + (root[2] + t) * 4;
                    if ((f->Bits >> t & 1) != 0 && tool[1] > f->Aux && g->Bytes[tool[0] + f->Aux] == c) candidates |= 1UL << t;
                }
                if (candidates == 0) return Reject;
                f->Bits = candidates;
                f->Aux++;
                return Consumed;
            }
            case 2:
                if (c != CallMid[f->Aux]) return Reject;
                if (++f->Aux == CallMid.Length)
                {
                    f->Phase = 3;
                    var argumentsNode = g->Tools[(root[2] + f->Aux2) * 4 + 2];
                    if (!Push(s, argumentsNode)) return Reject;
                }
                return Consumed;
            default:
                if (c != '}') return Reject;
                sink.CallCompleted();
                return Pop(s, Consumed);
        }
    }

    static ulong ToolMask(GrammarProgram* g) => g->ToolCount >= 64 ? ulong.MaxValue : (1UL << g->ToolCount) - 1;

    static int StepObject<TSink>(GrammarProgram* g, GrammarState* s, Frame* f, int* n, byte c, ref TSink sink) where TSink : struct, Tools.IGrammarSink
    {
        var count = n[1];
        var all = count >= 32 ? 0xFFFFFFFFUL : (1UL << count) - 1;
        switch (f->Phase)
        {
            case 0:
                if (c != '{') return Reject;
                f->Phase = 1;
                return Consumed;
            case 1:
                if (c == '}') return ((uint)f->Aux2 & (uint)n[3]) == (uint)n[3] ? Pop(s, Consumed) : Reject;
                goto case 5;
            case 5:
            {
                var candidates = ~(ulong)(uint)f->Aux2 & all;
                if (c != '"' || candidates == 0) return Reject;
                f->Phase = 2;
                f->Aux = 0;
                f->Bits = candidates;
                return Consumed;
            }
            case 2:
            {
                if (c == '"')
                {
                    for (var p = 0; p < count; p++)
                        if ((f->Bits >> p & 1) != 0 && g->Properties[(n[2] + p) * 4 + 1] == f->Aux)
                        {
                            f->Aux2 |= 1 << p;
                            f->Aux = p;
                            f->Phase = 3;
                            sink.PropertySelected(p);
                            return Consumed;
                        }
                    return Reject;
                }
                var candidates = 0UL;
                for (var p = 0; p < count; p++)
                {
                    var prop = g->Properties + (n[2] + p) * 4;
                    if ((f->Bits >> p & 1) != 0 && prop[1] > f->Aux && g->Bytes[prop[0] + f->Aux] == c) candidates |= 1UL << p;
                }
                if (candidates == 0) return Reject;
                f->Bits = candidates;
                f->Aux++;
                return Consumed;
            }
            case 3:
                if (c != ':') return Reject;
                f->Phase = 4;
                return Push(s, g->Properties[(n[2] + f->Aux) * 4 + 2]) ? Consumed : Reject;
            default:
                if (c == ',')
                {
                    if ((~(ulong)(uint)f->Aux2 & all) == 0) return Reject;
                    f->Phase = 5;
                    return Consumed;
                }
                return c == '}' && ((uint)f->Aux2 & (uint)n[3]) == (uint)n[3] ? Pop(s, Consumed) : Reject;
        }
    }

    static int StepFreeObject(GrammarState* s, Frame* f, byte c)
    {
        switch (f->Phase)
        {
            case 0:
                if (c != '{') return Reject;
                f->Phase = 1;
                return Consumed;
            case 1:
                if (c == '}') return Pop(s, Consumed);
                goto case 5;
            case 5:
                if (c != '"') return Reject;
                f->Phase = 3;
                return Push(s, Kind.StrNode) ? Reprocess : Reject;
            case 3:
                if (c != ':') return Reject;
                f->Phase = 4;
                return Push(s, Kind.AnyNode) ? Consumed : Reject;
            default:
                if (c == ',') { f->Phase = 5; return Consumed; }
                return c == '}' ? Pop(s, Consumed) : Reject;
        }
    }

    static int StepArray(GrammarState* s, Frame* f, int* n, byte c)
    {
        var max = n[3];
        switch (f->Phase)
        {
            case 0:
                if (c != '[') return Reject;
                f->Phase = 1;
                return Consumed;
            case 1:
                if (c == ']') return f->Aux >= n[2] ? Pop(s, Consumed) : Reject;
                goto case 3;
            case 3:
                if (max >= 0 && f->Aux >= max) return Reject;
                f->Aux++;
                f->Phase = 2;
                return Push(s, n[1]) ? Reprocess : Reject;
            default:
                if (c == ',') { f->Phase = 3; return max < 0 || f->Aux < max ? Consumed : Reject; }
                return c == ']' && f->Aux >= n[2] ? Pop(s, Consumed) : Reject;
        }
    }

    static int StepString<TSink>(Frame* f, int* n, GrammarState* s, byte c, ref TSink sink) where TSink : struct, Tools.IGrammarSink
    {
        var max = n[1];
        switch (f->Phase)
        {
            case 0:
                if (c != '"') return Reject;
                f->Phase = 1;
                return Consumed;
            case 1:
                if (c == '"')
                {
                    if (f->Aux < n[2]) return Reject;
                    sink.ValueCompleted();
                    return Pop(s, Consumed);
                }
                if (c == '\\')
                {
                    if (max >= 0 && f->Aux >= max) return Reject;
                    f->Phase = 2;
                    return Consumed;
                }
                if (c < 0x20) return Reject;
                if ((c & 0xC0) != 0x80)
                {
                    if (max >= 0 && f->Aux >= max) return Reject;
                    f->Aux++;
                }
                sink.StringByte(c);
                return Consumed;
            case 2:
                if (c == 'u') { f->Phase = 3; f->Aux2 = 0; f->Bits = 0; return Consumed; }
                if (c != '"' && c != '\\' && c != '/' && c != 'b' && c != 'f' && c != 'n' && c != 'r' && c != 't') return Reject;
                f->Aux++;
                f->Phase = 1;
                sink.StringByte(c switch
                {
                    (byte)'b' => (byte)'\b',
                    (byte)'f' => (byte)'\f',
                    (byte)'n' => (byte)'\n',
                    (byte)'r' => (byte)'\r',
                    (byte)'t' => (byte)'\t',
                    _ => c,
                });
                return Consumed;
            default:
                if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return Reject;
                f->Bits = f->Bits * 16 + HexValue(c);
                if (++f->Aux2 == 4)
                {
                    EmitUnit((int)f->Bits, ref sink);
                    f->Aux++;
                    f->Phase = 1;
                }
                return Consumed;
        }
    }

    static ulong HexValue(byte c) => c <= '9' ? (ulong)(c - '0') : (ulong)((c & 0xDF) - 'A' + 10);

    static void EmitUnit<TSink>(int unit, ref TSink sink) where TSink : struct, Tools.IGrammarSink
    {
        if (unit < 0x80) sink.StringByte((byte)unit);
        else if (unit < 0x800)
        {
            sink.StringByte((byte)(0xC0 | (unit >> 6)));
            sink.StringByte((byte)(0x80 | (unit & 0x3F)));
        }
        else if (unit is >= 0xD800 and <= 0xDFFF)
        {
            sink.StringByte(0xEF);
            sink.StringByte(0xBF);
            sink.StringByte(0xBD);
        }
        else
        {
            sink.StringByte((byte)(0xE0 | (unit >> 12)));
            sink.StringByte((byte)(0x80 | ((unit >> 6) & 0x3F)));
            sink.StringByte((byte)(0x80 | (unit & 0x3F)));
        }
    }

    static int StepNumber<TSink>(GrammarState* s, Frame* f, bool real, byte c, ref TSink sink) where TSink : struct, Tools.IGrammarSink
    {
        var digit = c >= '0' && c <= '9';
        switch (f->Phase)
        {
            case 0:
                if (c == '-') { f->Phase = 1; sink.NumberByte(c); return Consumed; }
                if (c == '0') { f->Phase = 2; sink.NumberByte(c); return Consumed; }
                if (digit) { f->Phase = 3; sink.NumberByte(c); return Consumed; }
                return Reject;
            case 1:
                if (c == '0') { f->Phase = 2; sink.NumberByte(c); return Consumed; }
                if (digit) { f->Phase = 3; sink.NumberByte(c); return Consumed; }
                return Reject;
            case 2:
            case 3:
                if (f->Phase == 3 && digit) { sink.NumberByte(c); return Consumed; }
                if (real && c == '.') { f->Phase = 4; sink.NumberByte(c); return Consumed; }
                if (real && (c == 'e' || c == 'E')) { f->Phase = 6; sink.NumberByte(c); return Consumed; }
                sink.ValueCompleted();
                return Pop(s, Reprocess);
            case 4:
                if (!digit) return Reject;
                f->Phase = 5;
                sink.NumberByte(c);
                return Consumed;
            case 5:
                if (digit) { sink.NumberByte(c); return Consumed; }
                if (c == 'e' || c == 'E') { f->Phase = 6; sink.NumberByte(c); return Consumed; }
                sink.ValueCompleted();
                return Pop(s, Reprocess);
            case 6:
                if (c == '+' || c == '-') { f->Phase = 7; sink.NumberByte(c); return Consumed; }
                goto case 7;
            case 7:
                if (!digit) return Reject;
                f->Phase = 8;
                sink.NumberByte(c);
                return Consumed;
            default:
                if (!digit)
                {
                    sink.ValueCompleted();
                    return Pop(s, Reprocess);
                }
                sink.NumberByte(c);
                return Consumed;
        }
    }

    static int StepBool<TSink>(GrammarState* s, Frame* f, byte c, ref TSink sink) where TSink : struct, Tools.IGrammarSink
    {
        if (f->Phase == 0)
        {
            if (c == 't') { f->Aux2 = 0; f->Aux = 1; f->Phase = 1; return Consumed; }
            if (c == 'f') { f->Aux2 = 1; f->Aux = 1; f->Phase = 1; return Consumed; }
            return Reject;
        }
        var word = f->Aux2 == 0 ? "true"u8 : "false"u8;
        if (c != word[f->Aux]) return Reject;
        if (++f->Aux != word.Length) return Consumed;
        sink.BoolSelected(f->Aux2 == 0);
        sink.ValueCompleted();
        return Pop(s, Consumed);
    }

    static int StepNull<TSink>(GrammarState* s, Frame* f, byte c, ref TSink sink) where TSink : struct, Tools.IGrammarSink
    {
        if (f->Phase == 0) f->Phase = 1;
        if (c != "null"u8[f->Aux]) return Reject;
        if (++f->Aux != 4) return Consumed;
        sink.NullSelected();
        sink.ValueCompleted();
        return Pop(s, Consumed);
    }

    static int StepEnum<TSink>(GrammarProgram* g, GrammarState* s, Frame* f, int* n, byte c, ref TSink sink) where TSink : struct, Tools.IGrammarSink
    {
        if (f->Phase == 0)
        {
            f->Phase = 1;
            f->Aux = 0;
            f->Bits = n[1] >= 64 ? ulong.MaxValue : (1UL << n[1]) - 1;
        }
        var candidates = 0UL;
        var complete = false;
        var completed = 0;
        for (var i = 0; i < n[1]; i++)
        {
            if ((f->Bits >> i & 1) == 0) continue;
            var literal = g->Literals + (n[2] + i) * 2;
            if (literal[1] == f->Aux) { complete = true; completed = i; }
            else if (g->Bytes[literal[0] + f->Aux] == c) candidates |= 1UL << i;
        }
        if (candidates == 0)
        {
            if (!complete) return Reject;
            sink.EnumSelected(completed);
            sink.ValueCompleted();
            return Pop(s, Reprocess);
        }
        f->Bits = candidates;
        f->Aux++;
        var longer = false;
        var exact = -1;
        for (var i = 0; i < n[1]; i++)
        {
            if ((candidates >> i & 1) == 0) continue;
            var length = g->Literals[(n[2] + i) * 2 + 1];
            if (length == f->Aux) exact = i; else longer = true;
        }
        if (exact < 0 || longer) return Consumed;
        sink.EnumSelected(exact);
        sink.ValueCompleted();
        return Pop(s, Consumed);
    }

    static int StepUnion(GrammarProgram* g, GrammarState* s, Frame* f, int* n, byte c)
    {
        for (var i = 0; i < n[1]; i++)
        {
            var branch = g->Unions[n[2] + i];
            if (!FirstByteOk(g, branch, c)) continue;
            *f = new Frame { Node = branch };
            return Reprocess;
        }
        return Reject;
    }

    static int StepAny(GrammarState* s, Frame* f, byte c)
    {
        int node;
        if (c == '"') node = Kind.StrNode;
        else if (c == '-' || (c >= '0' && c <= '9')) node = Kind.NumNode;
        else if (c == 't' || c == 'f') node = Kind.BoolNode;
        else if (c == 'n') node = Kind.NullNode;
        else if (c == '[') node = Kind.AnyArrayNode;
        else if (c == '{') node = Kind.AnyObjectNode;
        else return Reject;
        *f = new Frame { Node = node };
        return Reprocess;
    }

    // Appends the unique grammar-forced byte continuation to `destination`,
    // advancing `*state` past each forced byte. Stops when zero or more than
    // one byte is admissible, when the state completes, or when `destination`
    // fills. Returns the number of bytes written. W16 (jump-forward) consumes
    // this to skip decode steps over forced spans: each returned byte is the
    // only admissible continuation, so feeding it cannot change the outcome.
    public static int ForcedBytes(GrammarProgram* g, GrammarState* state, Span<byte> destination)
    {
        var sink = new Tools.NoSink();
        return ForcedBytes(g, state, destination, ref sink);
    }

    // Sink-reporting overload: forced bytes pass through `sink` (tool and
    // property events fire exactly as if each byte had arrived individually),
    // so typed decoding can fast-forward without losing events. Probing uses
    // a discard sink; sinks only observe and never affect admissibility.
    public static int ForcedBytes<TSink>(GrammarProgram* g, GrammarState* state, Span<byte> destination, ref TSink sink)
        where TSink : struct, Tools.IGrammarSink
    {
        var written = 0;
        var probe = new Tools.NoSink();
        while (written < destination.Length)
        {
            if (state->Depth == 0) break;
            var only = -1;
            var count = 0;
            for (var b = 0; b < 256; b++)
            {
                var copy = *state;
                if (!Advance(g, &copy, (byte)b, ref probe)) continue;
                count++;
                if (count > 1) break;
                only = b;
            }
            if (count != 1) break;
            if (!Advance(g, state, (byte)only, ref sink)) break;
            destination[written++] = (byte)only;
        }
        return written;
    }

    public void Dispose()
    {
        if (Program == null) return;
        Mem.Free(Program->Nodes);
        Mem.Free(Program->Properties);
        Mem.Free(Program->Literals);
        Mem.Free(Program->Unions);
        Mem.Free(Program->Tools);
        Mem.Free(Program->Bytes);
        Mem.Free(Program);
        Program = null;
    }
}
