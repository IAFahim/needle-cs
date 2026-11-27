using System;
using System.Buffers.Text;
using System.Diagnostics;
using System.Text;
using System.Threading;

namespace Needle;

/// <summary>What happened during a slice of a turn.</summary>
public enum TurnEventKind : byte
{
    /// <summary>The model thought some more; <see cref="TurnEvent.Text"/> is the new thinking.</summary>
    Thinking,
    /// <summary>The model is answering in words; <see cref="TurnEvent.Text"/> is the new text.</summary>
    Speaking,
    /// <summary>A call began and its tool is known (<see cref="TurnEvent.Name"/>).</summary>
    CallStarted,
    /// <summary>A call's arguments grew; <see cref="TurnEvent.Arguments"/> is the raw text so far.</summary>
    ArgumentProgress,
    /// <summary>
    /// A call closed. Its arguments are restored and repaired against the calls so far; it is provisional until the
    /// turn ends, because confidence, the floor and the whole-list repair are only known then.
    /// </summary>
    CallClosed,
    /// <summary>The turn ended with calls, text or a decision to call nothing; see <see cref="TurnEvent.Answer"/>.</summary>
    Done,
    /// <summary>The turn ended with its calls withheld (low confidence or ungrounded values).</summary>
    Refused,
    /// <summary>The turn could not finish; <see cref="Answer.Why"/> says why.</summary>
    Failed,
    /// <summary>The turn was cancelled.</summary>
    Cancelled,
}

internal struct TurnRecord
{
    public const byte Withheld = 1, Ungrounded = 2, Truncated = 4;

    public TurnEventKind Kind;
    public byte Flags;
    public int Call;
    public int Tool;
    public int Start;
    public int Length;
}

/// <summary>
/// The events of the agent's current turn, written by whichever thread runs the turn and read by the game thread.
/// Records and provisional bytes are appended and never moved; <see cref="Published"/> says how many a reader may see.
/// Records past the published mark may still be merged (several tokens of thinking in one slice become one event).
/// </summary>
internal sealed class TurnLog
{
    readonly TurnRecord[] records;
    readonly byte[] provisional;
    int count;
    int published;
    int provisionalLength;
    int reasoningSeen;
    int responseSeen;
    int progressCall;
    int progressLength;
    CallScanner scanner;

    public TurnLog(NeedleSession session, int reasoningTokens, int callTokens)
    {
        records = new TurnRecord[Math.Max(0, reasoningTokens) + 4 * Math.Max(1, callTokens) + 16];
        provisional = new byte[session.CallBufferBytes * 2 + 4096];
        session.EnsureStreamBuffers();
        Reset();
    }

    public TurnRecord[] Records => records;

    public byte[] Provisional => provisional;

    public int Published => Volatile.Read(ref published);

    public int Count => count;

    /// <summary>The game thread's read mark for <see cref="NeedleAgent.TryPoll"/>.</summary>
    public int Polled;

    public void Reset()
    {
        count = 0;
        Volatile.Write(ref published, 0);
        provisionalLength = 0;
        reasoningSeen = 0;
        responseSeen = 0;
        progressCall = -1;
        progressLength = 0;
        scanner = CallScanner.Create();
        Polled = 0;
    }

    public void Publish() => Volatile.Write(ref published, count);

    /// <summary>Turns what the last step produced into events.</summary>
    public void Observe(AgentState state)
    {
        var session = state.Session;
        var reasoning = session.TurnReasoningLength;
        if (reasoning > reasoningSeen)
        {
            Grow(TurnEventKind.Thinking, reasoningSeen, reasoning - reasoningSeen);
            reasoningSeen = reasoning;
        }
        var response = session.TurnResponseLength;
        if (response > responseSeen)
        {
            Grow(TurnEventKind.Speaking, responseSeen, response - responseSeen);
            responseSeen = response;
        }
        var text = session.TurnCallText;
        if (text.Length <= scanner.Position) return;
        for (;;)
        {
            var scan = scanner.Next(text, out var at);
            if (scan == CallScan.Named)
            {
                var tool = session.TryFindToolAlias(text.Slice(scanner.NameStart, scanner.NameEnd - scanner.NameStart), out var found) ? found : -1;
                Add(new TurnRecord { Kind = TurnEventKind.CallStarted, Call = scanner.Call, Tool = tool });
                continue;
            }
            Progress(text);
            if (scan == CallScan.None) return;
            Closed(session, at);
        }
    }

    void Progress(ReadOnlySpan<byte> text)
    {
        var length = scanner.ArgumentsLength(text);
        if (length == 0 || (scanner.Call == progressCall && length == progressLength)) return;
        var tool = CurrentTool();
        if (count > published && records[count - 1].Kind == TurnEventKind.ArgumentProgress && records[count - 1].Call == scanner.Call)
            records[count - 1].Length = length;
        else Add(new TurnRecord { Kind = TurnEventKind.ArgumentProgress, Call = scanner.Call, Tool = tool, Start = scanner.ArgumentsStart, Length = length });
        progressCall = scanner.Call;
        progressLength = length;
    }

    int CurrentTool()
    {
        for (var i = count - 1; i >= 0; i--)
            if (records[i].Kind == TurnEventKind.CallStarted && records[i].Call == scanner.Call) return records[i].Tool;
        return -1;
    }

    void Closed(NeedleSession session, int closeEnd)
    {
        var record = new TurnRecord { Kind = TurnEventKind.CallClosed, Call = scanner.Call, Tool = CurrentTool() };
        if (session.TryProvisionalCall(closeEnd, out var tool, out var arguments, out var withheld, out var ungrounded))
        {
            record.Tool = tool;
            if (withheld) record.Flags |= TurnRecord.Withheld;
            if (ungrounded) record.Flags |= TurnRecord.Ungrounded;
            if (provisionalLength + arguments.Length <= provisional.Length)
            {
                arguments.CopyTo(provisional.AsSpan(provisionalLength));
                record.Start = provisionalLength;
                record.Length = arguments.Length;
                provisionalLength += arguments.Length;
            }
            else record.Flags |= TurnRecord.Truncated;
        }
        else record.Flags |= TurnRecord.Truncated;
        Add(record);
    }

    void Grow(TurnEventKind kind, int start, int length)
    {
        if (count > published && records[count - 1].Kind == kind)
        {
            records[count - 1].Length += length;
            return;
        }
        Add(new TurnRecord { Kind = kind, Call = -1, Tool = -1, Start = start, Length = length });
    }

    void Add(in TurnRecord record)
    {
        // The last slot is kept for the turn's end, so a turn always reports how it ended.
        if (count >= records.Length - 1) return;
        records[count++] = record;
    }

    /// <summary>Adds the event that says how the turn ended, then publishes everything.</summary>
    public void Finish(AnswerKind kind)
    {
        var end = kind switch
        {
            AnswerKind.Calls or AnswerKind.Text or AnswerKind.Declined => TurnEventKind.Done,
            AnswerKind.Refused => TurnEventKind.Refused,
            AnswerKind.Cancelled => TurnEventKind.Cancelled,
            _ => TurnEventKind.Failed,
        };
        if (count < records.Length) records[count++] = new TurnRecord { Kind = end, Call = -1, Tool = -1 };
        Publish();
    }
}

internal sealed partial class AgentState
{
    public TurnLog? Log;

    /// <summary>True while a turn started with <see cref="NeedleAgent.Begin(string)"/> has not ended.</summary>
    public bool Streaming;

    public TurnLog OpenLog() => Log ??= new TurnLog(Session, Session.ReasoningBufferBytes / 64, Session.ResponseBufferBytes / 64);

    public void ThrowIfStreaming()
    {
        if (Streaming) throw new InvalidOperationException("a turn started with Begin is still running on this agent; Run it to the end or Cancel it first");
    }

    /// <summary>Ends an unfinished streamed turn as cancelled, before something else resets the conversation.</summary>
    public void AbandonStream()
    {
        if (!Streaming) return;
        Session.AbandonTurn();
        EndStream();
    }

    public void EndStream()
    {
        Streaming = false;
        Record(Session.TurnResult, Calls);
        Log!.Finish(Kind);
    }

    public TurnEvents RunSlice(int version, FrameBudget budget)
    {
        var log = Log!;
        var first = log.Polled;
        if (!Streaming)
        {
            log.Polled = log.Published;
            return new TurnEvents(this, version, first, log.Polled - first);
        }
        var session = Session;
        var started = Stopwatch.GetTimestamp();
        var work = session.TurnWork;
        var limit = budget.MaxMillisecondsPerSlice > 0 ? (long)(budget.MaxMillisecondsPerSlice * Stopwatch.Frequency / 1000.0) : long.MaxValue;
        for (;;)
        {
            var stepStarted = Stopwatch.GetTimestamp();
            var ended = session.StepStreamed(Calls, Reasoning, Response);
            log.Observe(this);
            if (ended)
            {
                EndStream();
                break;
            }
            var now = Stopwatch.GetTimestamp();
            if (budget.MaxTokensPerSlice > 0 && session.TurnWork - work >= budget.MaxTokensPerSlice) break;
            // Stop before a step that would likely overrun the slice, judging by the step just taken.
            if (limit != long.MaxValue && now - started + (now - stepStarted) > limit) break;
        }
        log.Publish();
        log.Polled = log.Published;
        return new TurnEvents(this, version, first, log.Polled - first);
    }
}

/// <summary>
/// One turn run a slice at a time on the caller's thread. Start it with <see cref="NeedleAgent.Begin(string)"/>,
/// call <see cref="Run(FrameBudget)"/> once a frame, and read <see cref="Answer"/> once <see cref="IsDone"/>. The
/// answer is the same, bit for bit, as <see cref="NeedleAgent.Ask(string)"/> gives for the same request. A turn is a
/// handle: it goes stale (and throws) once the agent starts another turn or is disposed.
/// </summary>
public readonly struct NeedleTurn : IEquatable<NeedleTurn>
{
    readonly int slot;
    readonly int generation;
    readonly int version;

    internal NeedleTurn(int slot, int generation, int version)
    {
        this.slot = slot;
        this.generation = generation;
        this.version = version;
    }

    AgentState State
    {
        get
        {
            var state = HandleTable<AgentState>.Get(slot, generation, nameof(NeedleAgent));
            if (state.Version != version) throw new InvalidOperationException("the agent has started another turn since this one; this turn is stale");
            return state;
        }
    }

    /// <summary>True once the turn has ended; <see cref="Answer"/> is then readable.</summary>
    public bool IsDone => !State.Streaming;

    /// <summary>
    /// Advances the turn by at most <paramref name="budget"/> and returns what happened in this slice. A token budget
    /// counts prompt tokens fed and tokens generated; a time budget stops before a step that would likely overrun it.
    /// A slice never splits a step: prefill runs in chunks of up to 64 tokens and every slice takes at least one step,
    /// so every call makes progress. An unlimited budget runs the turn to its end.
    /// </summary>
    public TurnEvents Run(FrameBudget budget) => State.RunSlice(version, budget);

    /// <summary>Advances the turn by at most this many tokens.</summary>
    public TurnEvents Run(int maxTokens) => State.RunSlice(version, FrameBudget.FromTokens(maxTokens));

    /// <summary>Advances the turn for at most about this long.</summary>
    public TurnEvents Run(TimeSpan slice) => State.RunSlice(version, FrameBudget.FromMilliseconds(slice.TotalMilliseconds));

    /// <summary>The finished answer. Throws while the turn is still running.</summary>
    public Answer Answer
    {
        get
        {
            var state = State;
            if (state.Streaming) throw new InvalidOperationException("the turn is still running; call Run until IsDone");
            return new Answer(state, version);
        }
    }

    /// <summary>Ends the turn now as cancelled. Like any cancelled turn, the agent forgets the conversation.</summary>
    public void Cancel()
    {
        if (!HandleTable<AgentState>.TryGet(slot, generation, out var state) || state.Version != version) return;
        state.AbandonStream();
    }

    public bool Equals(NeedleTurn other) => slot == other.slot && generation == other.generation && version == other.version;

    public override bool Equals(object? obj) => obj is NeedleTurn other && Equals(other);

    public override int GetHashCode() => (slot * 397) ^ generation ^ (version << 7);

    public static bool operator ==(NeedleTurn left, NeedleTurn right) => left.Equals(right);

    public static bool operator !=(NeedleTurn left, NeedleTurn right) => !left.Equals(right);
}

/// <summary>The events of one slice of a turn (or of one poll of a posted turn), in order. Valid until the agent's next turn.</summary>
public readonly ref struct TurnEvents
{
    readonly AgentState state;
    readonly int version;
    readonly int first;

    internal TurnEvents(AgentState state, int version, int first, int count)
    {
        this.state = state;
        this.version = version;
        this.first = first;
        Count = count;
    }

    public int Count { get; }

    public bool IsEmpty => Count == 0;

    public TurnEvent this[int index]
    {
        get
        {
            if ((uint)index >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(index));
            TurnEvent.Check(state, version);
            return new TurnEvent(state, version, first + index);
        }
    }

    /// <summary>The event that ended the turn, if it ended in this slice.</summary>
    public bool TryGetEnd(out TurnEvent end)
    {
        if (Count > 0 && this[Count - 1].IsFinal)
        {
            end = this[Count - 1];
            return true;
        }
        end = default;
        return false;
    }

    public Enumerator GetEnumerator() => new(this);

    public ref struct Enumerator
    {
        readonly TurnEvents events;
        int index;

        internal Enumerator(TurnEvents events)
        {
            this.events = events;
            index = -1;
        }

        public bool MoveNext() => ++index < events.Count;

        public TurnEvent Current => events[index];
    }
}

/// <summary>One event of a turn. Reads the agent's own buffers and throws once the agent starts another turn.</summary>
public readonly ref struct TurnEvent
{
    readonly AgentState state;
    readonly int version;
    readonly int index;

    internal TurnEvent(AgentState state, int version, int index)
    {
        this.state = state;
        this.version = version;
        this.index = index;
    }

    internal static void Check(AgentState state, int version)
    {
        if (state == null) throw new InvalidOperationException("this event is empty");
        if (state.Version != version)
        {
            if (state.Disposed) throw new ObjectDisposedException(nameof(NeedleAgent), "the agent behind this event has been disposed");
            throw new InvalidOperationException("the agent has started another turn since this event; read what you need before you ask again");
        }
    }

    TurnRecord Record
    {
        get
        {
            Check(state, version);
            return state.Log!.Records[index];
        }
    }

    public TurnEventKind Kind => Record.Kind;

    /// <summary>True for the event that ends the turn: Done, Refused, Failed or Cancelled.</summary>
    public bool IsFinal => Record.Kind >= TurnEventKind.Done;

    /// <summary>True until the turn ends: a closed call can still be changed or withheld by the final answer.</summary>
    public bool IsProvisional => Record.Kind < TurnEventKind.Done;

    /// <summary>The call this event is about, counted from 0 in the order the model wrote them; -1 when none.</summary>
    public int CallIndex => Record.Call;

    /// <summary>The tool's position in the tools JSON; -1 when none.</summary>
    public int Tool => Record.Tool;

    /// <summary>The tool's name, as written in the tools JSON; empty when the event is not about a call.</summary>
    public Utf8Span Name
    {
        get
        {
            var record = Record;
            return record.Tool < 0 || record.Kind is not (TurnEventKind.CallStarted or TurnEventKind.ArgumentProgress or TurnEventKind.CallClosed)
                ? default
                : new Utf8Span(state.ToolName(record.Tool));
        }
    }

    /// <summary>
    /// For <see cref="TurnEventKind.CallClosed"/>, the call's arguments restored and repaired against the calls so far.
    /// For <see cref="TurnEventKind.ArgumentProgress"/>, the raw arguments generated so far: an unfinished JSON object
    /// whose keys are the generated forms (snake_case) of the parameter names.
    /// </summary>
    public Utf8Span Arguments
    {
        get
        {
            var record = Record;
            return record.Kind switch
            {
                TurnEventKind.CallClosed => new Utf8Span(new ReadOnlySpan<byte>(state.Log!.Provisional, record.Start, record.Length)),
                TurnEventKind.ArgumentProgress => new Utf8Span(new ReadOnlySpan<byte>(state.Session.CallTextBuffer, record.Start, record.Length)),
                _ => default,
            };
        }
    }

    /// <summary>For Thinking and Speaking, the text added by this event.</summary>
    public Utf8Span Text
    {
        get
        {
            var record = Record;
            return record.Kind switch
            {
                TurnEventKind.Thinking => new Utf8Span(new ReadOnlySpan<byte>(state.Reasoning, record.Start, record.Length)),
                TurnEventKind.Speaking => new Utf8Span(new ReadOnlySpan<byte>(state.Response, record.Start, record.Length)),
                _ => default,
            };
        }
    }

    /// <summary>For Thinking and Speaking, everything said so far, this event included.</summary>
    public Utf8Span TextSoFar
    {
        get
        {
            var record = Record;
            return record.Kind switch
            {
                TurnEventKind.Thinking => new Utf8Span(new ReadOnlySpan<byte>(state.Reasoning, 0, record.Start + record.Length)),
                TurnEventKind.Speaking => new Utf8Span(new ReadOnlySpan<byte>(state.Response, 0, record.Start + record.Length)),
                _ => default,
            };
        }
    }

    /// <summary>A closed call that repair, with the calls so far, would drop (negated, out of range or a repeat).</summary>
    public bool IsWithheld => (Record.Flags & TurnRecord.Withheld) != 0;

    /// <summary>A closed call with a value the grounding gate cannot find in the request; the turn will likely refuse it.</summary>
    public bool IsUngrounded => (Record.Flags & TurnRecord.Ungrounded) != 0;

    /// <summary>The turn's answer, for the final event.</summary>
    public Answer Answer
    {
        get
        {
            if (!IsFinal) throw new InvalidOperationException("only the final event of a turn carries its answer");
            return new Answer(state, version);
        }
    }

    public bool Is(string name) => Name.Equals(name);

    public bool TryGet(string name, out Utf8Span value)
    {
        var found = JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out _);
        value = found ? new Utf8Span(raw) : default;
        return found;
    }

    public bool TryGetInt32(string name, out int value)
    {
        value = 0;
        return JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out var isString) && !isString && Utf8Parser.TryParse(raw, out value, out var used) && used == raw.Length;
    }

    public bool TryGetDouble(string name, out double value)
    {
        value = 0;
        return JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out var isString) && !isString && Utf8Parser.TryParse(raw, out value, out var used) && used == raw.Length;
    }

    public bool TryGetText(string name, Span<byte> destination, out int written)
    {
        written = 0;
        if (!JsonArgs.TryFind(Arguments.Bytes, name.AsSpan(), out var raw, out var isString) || !isString) return false;
        JsonArgs.UnescapeTruncating(raw, destination, out written);
        return true;
    }

    public override string ToString()
    {
        if (state == null) return "(empty event)";
        var record = Record;
        return record.Kind switch
        {
            TurnEventKind.Thinking or TurnEventKind.Speaking => record.Kind + ": " + Text.ToString(),
            TurnEventKind.CallStarted => "CallStarted #" + record.Call + " " + Name.ToString(),
            TurnEventKind.ArgumentProgress => "ArgumentProgress #" + record.Call + " " + Name.ToString() + " " + Arguments.ToString(),
            TurnEventKind.CallClosed => "CallClosed #" + record.Call + " " + Name.ToString() + "(" + Arguments.ToString() + ")" + (IsWithheld ? " withheld" : ""),
            _ => record.Kind + ": " + Answer.ToString(),
        };
    }
}

public readonly partial struct NeedleAgent
{
    /// <summary>
    /// Starts a turn that runs a slice at a time on the caller's thread with <see cref="NeedleTurn.Run(FrameBudget)"/>.
    /// Nothing runs until the first slice. The request is copied, so the caller's buffer is free on return.
    /// </summary>
    public NeedleTurn Begin(ReadOnlySpan<byte> utf8) => BeginTurn(utf8, false);

    public NeedleTurn Begin(ReadOnlySpan<char> text) => BeginTurn(text, false);

    public NeedleTurn Begin(string text) => BeginTurn((text ?? "").AsSpan(), false);

    /// <summary>Hands a tool's result back as a streamed turn, the sliced version of <see cref="Continue(string)"/>.</summary>
    public NeedleTurn BeginContinue(ReadOnlySpan<byte> resultJsonUtf8) => BeginTurn(resultJsonUtf8, true);

    public NeedleTurn BeginContinue(string resultJson) => BeginTurn((resultJson ?? "").AsSpan(), true);

    NeedleTurn BeginTurn(ReadOnlySpan<char> text, bool toolResult)
    {
        var state = State;
        state.ThrowIfBusy();
        state.ThrowIfStreaming();
        var count = Encoding.UTF8.GetByteCount(text);
        if (count > state.Input.Length) return BeginTurn(state, default, toolResult, FailureReason.InputTooLong);
        var written = count == 0 ? 0 : Encoding.UTF8.GetBytes(text, state.Input);
        return BeginTurn(state, new ReadOnlySpan<byte>(state.Input, 0, written), toolResult, FailureReason.None);
    }

    NeedleTurn BeginTurn(ReadOnlySpan<byte> utf8, bool toolResult)
    {
        var state = State;
        state.ThrowIfBusy();
        state.ThrowIfStreaming();
        return BeginTurn(state, utf8, toolResult, FailureReason.None);
    }

    NeedleTurn BeginTurn(AgentState state, ReadOnlySpan<byte> utf8, bool toolResult, FailureReason early)
    {
        var log = state.OpenLog();
        state.Invalidate();
        log.Reset();
        if (early != FailureReason.None)
        {
            state.Record(NeedleSession.FailedBeforeRun(early), state.Calls);
            log.Finish(state.Kind);
        }
        else if (state.Session.BeginStreamed(utf8, toolResult)) state.Streaming = true;
        else
        {
            state.Record(state.Session.TurnResult, state.Calls);
            log.Finish(state.Kind);
        }
        return new NeedleTurn(slot, generation, state.Version);
    }

    /// <summary>
    /// Reads what a posted request (<see cref="TryPost(string)"/>) has done since the last poll, without blocking and
    /// without computing on the caller's thread. True when there are new events. The last event of a finished request
    /// is its end; take the answer with <see cref="TryTake"/> as before.
    /// </summary>
    public bool TryPoll(out TurnEvents events)
    {
        var state = State;
        var log = state.Log;
        if (log == null || state.Mailbox == null)
        {
            events = default;
            return false;
        }
        var published = log.Published;
        var from = log.Polled;
        events = new TurnEvents(state, state.Version, from, published - from);
        log.Polled = published;
        return published > from;
    }
}
