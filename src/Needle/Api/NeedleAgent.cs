using System;
using System.Buffers;
using System.Text;

namespace Needle;

internal sealed class AgentState
{
    public readonly NeedleSession Session;
    public readonly ModelState Model;
    public readonly byte[] Input;
    public readonly byte[] Calls;
    public readonly byte[] Reasoning;
    public readonly byte[] Response;
    public int Version;
    public bool Disposed;
    public Reply Last;
    public bool HasAnswer;
    public byte[] CallSource;
    public AnswerKind Kind;
    public AnswerWhy Why;

    public AgentState(ModelState model, string toolsJson, in AgentOptions options)
    {
        Model = model;
        Session = new NeedleSession(model.Model, toolsJson, options.ToNeedleOptions(null));
        Input = new byte[Session.MaxInputBytes];
        Calls = new byte[Session.CallBufferBytes];
        Reasoning = new byte[Session.ReasoningBufferBytes];
        Response = new byte[Session.ResponseBufferBytes];
        CallSource = Calls;
    }

    public void Record(in Reply reply, byte[] callSource)
    {
        Last = reply;
        HasAnswer = true;
        CallSource = callSource;
        (Kind, Why) = Classify(reply);
    }

    (AnswerKind, AnswerWhy) Classify(in Reply reply)
    {
        switch (reply.Status)
        {
            case ReplyStatus.Called: return (AnswerKind.Calls, AnswerWhy.None);
            case ReplyStatus.Responded: return (AnswerKind.Text, AnswerWhy.None);
            case ReplyStatus.Declined: return (AnswerKind.Declined, AnswerWhy.Declined);
            case ReplyStatus.Cancelled: return (AnswerKind.Cancelled, AnswerWhy.Cancelled);
            case ReplyStatus.Refused: return (AnswerKind.Refused, Session.LastWasUngrounded ? AnswerWhy.Ungrounded : AnswerWhy.LowConfidence);
        }
        if (reply.Failure == FailureReason.CallsBufferTooSmall)
        {
            CallSource = Session.RestoredCalls;
            if (Session.LastWasUngrounded) return (AnswerKind.Refused, AnswerWhy.Ungrounded);
            if (reply.Confidence < Session.RefuseFloor) return (AnswerKind.Refused, AnswerWhy.LowConfidence);
            return (AnswerKind.Calls, AnswerWhy.None);
        }
        return (AnswerKind.Failed, reply.Failure switch
        {
            FailureReason.EmptyInput => AnswerWhy.EmptyInput,
            FailureReason.InputTooLong => AnswerWhy.InputTooLong,
            FailureReason.ContextExhausted => AnswerWhy.ContextFull,
            FailureReason.CallTokenBudgetExceeded or FailureReason.CallTextTooLong => AnswerWhy.CallTooLong,
            FailureReason.GrammarDeadEnd or FailureReason.GrammarRejectedToken => AnswerWhy.GrammarDeadEnd,
            FailureReason.MalformedCalls => AnswerWhy.MalformedCalls,
            FailureReason.ToolContextFailed => AnswerWhy.ToolContextFailed,
            FailureReason.FrameBudgetExceeded => AnswerWhy.FrameBudget,
            _ => AnswerWhy.Failed,
        });
    }

    public void Invalidate()
    {
        Version = Version == int.MaxValue ? 1 : Version + 1;
        HasAnswer = false;
    }
}

/// <summary>
/// A conversation with a Needle model about a fixed set of tools. Ask in plain text, get calls back.
/// An agent is a handle: copying it copies the handle, not the conversation, and a disposed or
/// <c>default</c> agent throws <see cref="ObjectDisposedException"/>. One agent serves one thread at a time.
/// </summary>
public readonly struct NeedleAgent : IDisposable, IEquatable<NeedleAgent>
{
    readonly int slot;
    readonly int generation;

    NeedleAgent(int slot, int generation)
    {
        this.slot = slot;
        this.generation = generation;
    }

    /// <summary>Loads the model and makes an agent that owns it. Disposing the agent frees the model.</summary>
    public static NeedleAgent Open(string modelPath, string toolsJson, in AgentOptions options = default)
    {
        var model = new ModelState(Model.Open(modelPath));
        try
        {
            return Create(model, toolsJson, options);
        }
        finally
        {
            model.Release();
        }
    }

    internal static NeedleAgent Create(ModelState model, string toolsJson, in AgentOptions options)
    {
        if (toolsJson == null) throw new ArgumentNullException(nameof(toolsJson));
        model.AddReference();
        AgentState state;
        try
        {
            state = new AgentState(model, toolsJson, options);
        }
        catch
        {
            model.Release();
            throw;
        }
        HandleTable<AgentState>.Add(state, out var slot, out var generation);
        return new NeedleAgent(slot, generation);
    }

    public bool IsOpen => HandleTable<AgentState>.TryGet(slot, generation, out _);

    AgentState State => HandleTable<AgentState>.Get(slot, generation, nameof(NeedleAgent));

    /// <summary>The engine-level session, for APIs that still take one.</summary>
    public NeedleSession Session => State.Session;

    /// <summary>The longest request, in UTF-8 bytes, that fits.</summary>
    public int MaxInputBytes => State.Session.MaxInputBytes;

    /// <summary>Asks in plain text. The answer stays readable until the next call on this agent.</summary>
    public Answer Ask(ReadOnlySpan<byte> utf8) => Turn(State, utf8, false);

    public Answer Ask(ReadOnlySpan<char> text) => Turn(State, text, false);

    public Answer Ask(string text) => Turn(State, (text ?? "").AsSpan(), false);

    /// <summary>Hands a tool's result back to the model and gets its next step.</summary>
    public Answer Continue(ReadOnlySpan<byte> resultJsonUtf8) => Turn(State, resultJsonUtf8, true);

    public Answer Continue(ReadOnlySpan<char> resultJson) => Turn(State, resultJson, true);

    public Answer Continue(string resultJson) => Turn(State, (resultJson ?? "").AsSpan(), true);

    /// <summary>
    /// Asks and writes the calls as a JSON array into <paramref name="callsJson"/>, which the caller owns.
    /// <see cref="OperationStatus.DestinationTooSmall"/> means the buffer was too small; <see cref="OperationStatus.InvalidData"/>
    /// means the turn failed (see <paramref name="details"/>). Refused and text answers are <see cref="OperationStatus.Done"/>.
    /// </summary>
    public OperationStatus Ask(ReadOnlySpan<byte> utf8, Span<byte> callsJson, out Reply details)
    {
        var state = State;
        state.Invalidate();
        details = state.Session.Complete(utf8, callsJson);
        if (details.Status != ReplyStatus.Failed) return OperationStatus.Done;
        return details.Failure == FailureReason.CallsBufferTooSmall ? OperationStatus.DestinationTooSmall : OperationStatus.InvalidData;
    }

    /// <summary>Replaces the system facts (who the user is, the scene, today's date) and starts a new conversation.</summary>
    public void SetSystem(string? facts)
    {
        var state = State;
        state.Invalidate();
        state.Session.SetSystem(facts);
    }

    public void SetSystem(ReadOnlySpan<byte> factsUtf8) => SetSystem(factsUtf8.IsEmpty ? null : Encoding.UTF8.GetString(factsUtf8));

    /// <summary>Forgets the conversation but keeps the tools and the system facts.</summary>
    public void Forget()
    {
        var state = State;
        state.Invalidate();
        state.Session.Reset();
    }

    /// <summary>Stops a turn running on another thread as soon as it can. Safe to call from any thread.</summary>
    public void Cancel()
    {
        if (HandleTable<AgentState>.TryGet(slot, generation, out var state)) state.Session.RequestCancel();
    }

    /// <summary>Runs the kernels once so the first real turn is not the slow one.</summary>
    public void WarmUp() => State.Session.WarmUp();

    public void Dispose()
    {
        var state = HandleTable<AgentState>.Remove(slot, generation);
        if (state == null) return;
        state.Invalidate();
        state.Disposed = true;
        try
        {
            state.Session.Dispose();
        }
        finally
        {
            state.Model.Release();
        }
    }

    static Answer Turn(AgentState state, ReadOnlySpan<char> text, bool toolResult)
    {
        var count = Encoding.UTF8.GetByteCount(text);
        if (count > state.Input.Length)
        {
            state.Invalidate();
            state.Record(NeedleSession.FailedBeforeRun(FailureReason.InputTooLong), state.Calls);
            return new Answer(state, state.Version);
        }
        var written = count == 0 ? 0 : Encoding.UTF8.GetBytes(text, state.Input);
        return Turn(state, new ReadOnlySpan<byte>(state.Input, 0, written), toolResult);
    }

    static Answer Turn(AgentState state, ReadOnlySpan<byte> utf8, bool toolResult)
    {
        state.Invalidate();
        var reply = toolResult
            ? state.Session.CompleteWithToolResult(utf8, state.Calls, state.Reasoning, state.Response)
            : state.Session.Complete(utf8, state.Calls, state.Reasoning, state.Response);
        state.Record(reply, state.Calls);
        return new Answer(state, state.Version);
    }

    public bool Equals(NeedleAgent other) => slot == other.slot && generation == other.generation;

    public override bool Equals(object? obj) => obj is NeedleAgent other && Equals(other);

    public override int GetHashCode() => (slot * 397) ^ generation;

    public static bool operator ==(NeedleAgent left, NeedleAgent right) => left.Equals(right);

    public static bool operator !=(NeedleAgent left, NeedleAgent right) => !left.Equals(right);
}
