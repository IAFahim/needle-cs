using System;
using System.Buffers.Text;

namespace Needle;

/// <summary>What the model did with a request.</summary>
public enum AnswerKind : byte
{
    /// <summary>It called one or more tools.</summary>
    Calls,
    /// <summary>It answered in words instead of calling a tool.</summary>
    Text,
    /// <summary>It produced calls, but they were withheld; they are in <see cref="Answer.Suppressed"/>.</summary>
    Refused,
    /// <summary>It chose to call nothing.</summary>
    Declined,
    /// <summary>The turn could not finish; <see cref="Answer.Why"/> says why.</summary>
    Failed,
    /// <summary><see cref="NeedleAgent.Cancel"/> stopped the turn.</summary>
    Cancelled,
}

/// <summary>Why an answer is not <see cref="Answer.Ok"/>.</summary>
public enum AnswerWhy : byte
{
    None,
    /// <summary>Confidence fell below the floor (<see cref="AgentOptions.RefuseBelow"/>).</summary>
    LowConfidence,
    /// <summary>A value in the calls does not appear in the request (the 4.x grounding gate).</summary>
    Ungrounded,
    Declined,
    Cancelled,
    EmptyInput,
    InputTooLong,
    ContextFull,
    CallTooLong,
    GrammarDeadEnd,
    MalformedCalls,
    ToolContextFailed,
    FrameBudget,
    Failed,
}

/// <summary>
/// The model's answer to one request. It reads the agent's own buffers, so it costs nothing, and it is only
/// valid until the next call on the same agent; reading it after that throws <see cref="InvalidOperationException"/>.
/// <c>foreach</c> over an answer walks its calls.
/// </summary>
public readonly ref struct Answer
{
    readonly AgentState state;
    readonly int version;

    internal Answer(AgentState state, int version)
    {
        this.state = state;
        this.version = version;
    }

    AgentState Live
    {
        get
        {
            if (state == null) throw new InvalidOperationException("this answer is empty; get one from NeedleAgent.Ask");
            if (state.Version != version || !state.HasAnswer)
            {
                if (state.Disposed) throw new ObjectDisposedException(nameof(NeedleAgent), "the agent behind this answer has been disposed");
                throw new InvalidOperationException("a later call on the same agent replaced this answer; read what you need before you ask again");
            }
            return state;
        }
    }

    public AnswerKind Kind => Live.Kind;

    /// <summary>True when there is something to act on: calls or text.</summary>
    public bool Ok => Live.Kind is AnswerKind.Calls or AnswerKind.Text;

    /// <summary>Why the answer is not <see cref="Ok"/>; <see cref="AnswerWhy.None"/> when it is.</summary>
    public AnswerWhy Why => Live.Why;

    /// <summary>The calls to run, in order. Empty unless <see cref="Kind"/> is <see cref="AnswerKind.Calls"/>.</summary>
    public CallList Calls
    {
        get
        {
            var live = Live;
            return new CallList(live, version, live.Kind == AnswerKind.Calls ? live.Session.ViewCallCount : 0);
        }
    }

    /// <summary>Calls the model made but that were withheld (4.x <c>suppressed_calls</c>).</summary>
    public CallList Suppressed
    {
        get
        {
            var live = Live;
            return new CallList(live, version, live.Kind == AnswerKind.Refused ? live.Session.ViewCallCount : 0);
        }
    }

    /// <summary>What the model said when it answered in words.</summary>
    public Utf8Span Text
    {
        get
        {
            var live = Live;
            return new Utf8Span(Trim(new ReadOnlySpan<byte>(live.Response, 0, live.Last.ResponseLength)));
        }
    }

    /// <summary>What the model thought before it answered.</summary>
    public Utf8Span Thinking
    {
        get
        {
            var live = Live;
            return new Utf8Span(Trim(new ReadOnlySpan<byte>(live.Reasoning, 0, live.Last.ReasoningLength)));
        }
    }

    /// <summary>Combined confidence from 0 to 1.</summary>
    public float Confidence => Live.Last.Confidence;

    /// <summary>Values that the grounding gate could not find in the request, one per line.</summary>
    public Utf8Span Ungrounded => new(Live.Session.Ungrounded);

    /// <summary>Everything else: repair flags, both confidences, token counts and timings.</summary>
    public Reply Details => Live.Last;

    public CallList.Enumerator GetEnumerator() => Calls.GetEnumerator();

    public override string ToString()
    {
        if (state == null) return "(empty answer)";
        var live = Live;
        return live.Kind switch
        {
            AnswerKind.Calls => $"{Calls.Count} call(s), confidence {live.Last.Confidence:0.00}",
            AnswerKind.Text => Text.ToString(),
            _ => $"{live.Kind}: {live.Why}",
        };
    }

    static ReadOnlySpan<byte> Trim(ReadOnlySpan<byte> text)
    {
        var start = 0;
        var end = text.Length;
        while (start < end && IsSpace(text[start])) start++;
        while (end > start && IsSpace(text[end - 1])) end--;
        return text.Slice(start, end - start);
    }

    static bool IsSpace(byte b) => b is (byte)' ' or (byte)'\n' or (byte)'\r' or (byte)'\t';
}

/// <summary>The calls of one answer.</summary>
public readonly ref struct CallList
{
    readonly AgentState state;
    readonly int version;

    internal CallList(AgentState state, int version, int count)
    {
        this.state = state;
        this.version = version;
        Count = count;
    }

    public int Count { get; }

    public bool IsEmpty => Count == 0;

    public Call this[int index]
    {
        get
        {
            if ((uint)index >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(index));
            return new Call(state, version, index);
        }
    }

    public Enumerator GetEnumerator() => new(this);

    public ref struct Enumerator
    {
        readonly CallList list;
        int index;

        internal Enumerator(CallList list)
        {
            this.list = list;
            index = -1;
        }

        public bool MoveNext() => ++index < list.Count;

        public Call Current => list[index];
    }
}

/// <summary>One tool call: the tool's name and its arguments as JSON, after 4.x repair.</summary>
public readonly ref struct Call
{
    readonly AgentState state;
    readonly int version;

    internal Call(AgentState state, int version, int index)
    {
        this.state = state;
        this.version = version;
        Index = index;
    }

    public int Index { get; }

    CallSpan Span
    {
        get
        {
            if (state.Version != version || !state.HasAnswer)
                throw new InvalidOperationException("a later call on the same agent replaced this call; read what you need before you ask again");
            return state.Session.ViewCall(Index);
        }
    }

    /// <summary>The tool's name, as written in the tools JSON.</summary>
    public Utf8Span Name => new(state.Session.ToolNameUtf8(Span.Tool));

    /// <summary>The position of the tool in the tools JSON.</summary>
    public int Tool => Span.Tool;

    /// <summary>The arguments as a JSON object.</summary>
    public Utf8Span Arguments
    {
        get
        {
            var span = Span;
            return new Utf8Span(new ReadOnlySpan<byte>(state.CallSource, span.ArgumentsStart, span.ArgumentsLength));
        }
    }

    public bool Is(string name) => Name.Equals(name);

    public bool Is(ReadOnlySpan<byte> nameUtf8) => Name.Equals(nameUtf8);

    /// <summary>Finds one argument. Strings come without their quotes and with escapes as written; see <see cref="TryGetString(string, Span{byte}, out int)"/>.</summary>
    public bool TryGet(string name, out Utf8Span value) => Found(JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out _), raw, out value);

    public bool TryGet(ReadOnlySpan<byte> nameUtf8, out Utf8Span value) => Found(JsonArgs.TryFind(Arguments.Bytes, nameUtf8, out var raw, out _), raw, out value);

    public bool TryGetInt32(string name, out int value)
    {
        value = 0;
        return JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out var isString) && !isString && Utf8Parser.TryParse(raw, out value, out var used) && used == raw.Length;
    }

    public bool TryGetInt64(string name, out long value)
    {
        value = 0;
        return JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out var isString) && !isString && Utf8Parser.TryParse(raw, out value, out var used) && used == raw.Length;
    }

    public bool TryGetDouble(string name, out double value)
    {
        value = 0;
        return JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out var isString) && !isString && Utf8Parser.TryParse(raw, out value, out var used) && used == raw.Length;
    }

    public bool TryGetBoolean(string name, out bool value)
    {
        value = false;
        if (!JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out var isString) || isString) return false;
        if (raw.SequenceEqual("true"u8)) { value = true; return true; }
        return raw.SequenceEqual("false"u8);
    }

    /// <summary>Copies a string argument into <paramref name="destination"/> with its JSON escapes decoded.</summary>
    public bool TryGetString(string name, Span<byte> destination, out int written)
    {
        written = 0;
        return JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out var isString) && isString && JsonArgs.TryUnescape(raw, destination, out written);
    }

    /// <summary>A string argument as a .NET string, escapes decoded. Allocates.</summary>
    public string? GetString(string name)
    {
        if (!JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out var isString) || !isString) return null;
        var buffer = new byte[raw.Length];
        return JsonArgs.TryUnescape(raw, buffer, out var written) ? System.Text.Encoding.UTF8.GetString(buffer, 0, written) : null;
    }

    public override string ToString() => Name.ToString() + "(" + Arguments.ToString() + ")";

    static bool Found(bool found, ReadOnlySpan<byte> raw, out Utf8Span value)
    {
        value = found ? new Utf8Span(raw) : default;
        return found;
    }
}
