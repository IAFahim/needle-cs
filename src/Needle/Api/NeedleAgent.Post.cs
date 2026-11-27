using System;
using System.Text;
using System.Threading;

namespace Needle;

internal sealed partial class AgentState
{
    public AgentMailbox? Mailbox;

    public AgentMailbox OpenMailbox()
    {
        if (Mailbox != null) return Mailbox;
        ThrowIfStreaming();
        OpenLog();
        return Mailbox = new AgentMailbox(this);
    }

    public void StopMailbox() => Mailbox?.Stop();
}

/// <summary>
/// One request in flight per agent, run on the agent's own thread. The caller never blocks: it posts, keeps
/// rendering, and takes the answer on a later frame. The thread sleeps on an event while idle, so a post wakes
/// it at once rather than after a sleep quantum, and it uses no CPU between requests.
/// </summary>
internal sealed class AgentMailbox
{
    const int Idle = 0, Queued = 1, Running = 2, Done = 3;

    readonly AgentState state;
    readonly Thread thread;
    readonly ManualResetEventSlim wake = new(false);
    readonly ManualResetEventSlim finished = new(true);
    int phase;
    int length;
    bool toolResult;
    volatile bool stopping;
    volatile bool cancelQueued;

    public AgentMailbox(AgentState state)
    {
        this.state = state;
        thread = new Thread(Loop) { IsBackground = true, Name = "Needle agent" };
        thread.Start();
    }

    public bool Busy => Volatile.Read(ref phase) is Queued or Running;

    public bool HasResult => Volatile.Read(ref phase) == Done;

    public bool TryPost(ReadOnlySpan<char> text, bool asToolResult)
    {
        if (Busy) return false;
        state.ThrowIfStreaming();
        var count = Encoding.UTF8.GetByteCount(text);
        if (count > state.Input.Length)
        {
            state.Invalidate();
            state.Log!.Reset();
            state.Record(NeedleSession.FailedBeforeRun(FailureReason.InputTooLong), state.Calls);
            state.Log.Finish(state.Kind);
            Volatile.Write(ref phase, Done);
            return true;
        }
        var written = count == 0 ? 0 : Encoding.UTF8.GetBytes(text, state.Input);
        return Enqueue(written, asToolResult);
    }

    public bool TryPost(ReadOnlySpan<byte> utf8, bool asToolResult)
    {
        if (Busy) return false;
        state.ThrowIfStreaming();
        if (utf8.Length > state.Input.Length)
        {
            state.Invalidate();
            state.Log!.Reset();
            state.Record(NeedleSession.FailedBeforeRun(FailureReason.InputTooLong), state.Calls);
            state.Log.Finish(state.Kind);
            Volatile.Write(ref phase, Done);
            return true;
        }
        utf8.CopyTo(state.Input);
        return Enqueue(utf8.Length, asToolResult);
    }

    /// <summary>Cancels a request that has not started yet; a running one is cancelled through the session.</summary>
    public void Cancel() => cancelQueued = true;

    bool Enqueue(int bytes, bool asToolResult)
    {
        state.Invalidate();
        state.Log!.Reset();
        cancelQueued = false;
        length = bytes;
        toolResult = asToolResult;
        finished.Reset();
        Volatile.Write(ref phase, Queued);
        wake.Set();
        return true;
    }

    /// <summary>Marks the result as taken so the next post can run. Returns false while the request is still running.</summary>
    public bool TryTake()
    {
        if (Volatile.Read(ref phase) != Done) return false;
        Volatile.Write(ref phase, Idle);
        return true;
    }

    public void WaitForResult() => finished.Wait();

    public void Stop()
    {
        if (stopping) return;
        stopping = true;
        state.Session.RequestCancel();
        wake.Set();
        if (Thread.CurrentThread != thread) thread.Join();
        wake.Dispose();
        finished.Dispose();
    }

    void Loop()
    {
        while (true)
        {
            wake.Wait();
            wake.Reset();
            if (stopping) return;
            if (Volatile.Read(ref phase) != Queued) continue;
            Volatile.Write(ref phase, Running);
            Reply reply;
            var log = state.Log!;
            if (cancelQueued)
            {
                reply = new Reply(ReplyStatus.Cancelled, FailureReason.None, ReplyFlags.None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                state.Record(reply, state.Calls);
                log.Finish(state.Kind);
                Volatile.Write(ref phase, Done);
                finished.Set();
                continue;
            }
            try
            {
                // The same steps Ask runs, one at a time, so the game thread can poll progress between them.
                var session = state.Session;
                if (session.BeginStreamed(new ReadOnlySpan<byte>(state.Input, 0, length), toolResult))
                    while (!session.StepStreamed(state.Calls, state.Reasoning, state.Response))
                    {
                        log.Observe(state);
                        log.Publish();
                    }
                log.Observe(state);
                reply = session.TurnResult;
            }
            catch
            {
                reply = NeedleSession.FailedBeforeRun(FailureReason.None);
            }
            state.Record(reply, state.Calls);
            log.Finish(state.Kind);
            Volatile.Write(ref phase, Done);
            finished.Set();
            if (stopping) return;
        }
    }
}

public readonly partial struct NeedleAgent
{
    /// <summary>
    /// Starts a request on the agent's own thread and returns at once. False when one is already running.
    /// Take the answer with <see cref="TryTake"/> on a later frame. Copies the text, so the caller's buffer is free on return.
    /// </summary>
    public bool TryPost(string text) => State.OpenMailbox().TryPost((text ?? "").AsSpan(), false);

    public bool TryPost(ReadOnlySpan<char> text) => State.OpenMailbox().TryPost(text, false);

    public bool TryPost(ReadOnlySpan<byte> utf8) => State.OpenMailbox().TryPost(utf8, false);

    /// <summary>Like <see cref="TryPost(string)"/>, but throws if a request is already running.</summary>
    public void Post(string text)
    {
        if (!TryPost(text)) throw new InvalidOperationException("a posted request is still running on this agent; TryTake its answer first");
    }

    public void Post(ReadOnlySpan<char> text)
    {
        if (!TryPost(text)) throw new InvalidOperationException("a posted request is still running on this agent; TryTake its answer first");
    }

    public void Post(ReadOnlySpan<byte> utf8)
    {
        if (!TryPost(utf8)) throw new InvalidOperationException("a posted request is still running on this agent; TryTake its answer first");
    }

    /// <summary>Posts a tool's result, the background version of <see cref="Continue(string)"/>.</summary>
    public bool TryPostContinue(string resultJson) => State.OpenMailbox().TryPost((resultJson ?? "").AsSpan(), true);

    public bool TryPostContinue(ReadOnlySpan<byte> resultJsonUtf8) => State.OpenMailbox().TryPost(resultJsonUtf8, true);

    /// <summary>True while a posted request is queued or running.</summary>
    public bool IsBusy => State.Mailbox is { } mailbox && mailbox.Busy;

    /// <summary>
    /// Takes the answer of the last posted request once it is ready, without blocking. The answer stays readable until
    /// the next post or ask on this agent.
    /// </summary>
    public bool TryTake(out Answer answer)
    {
        var state = State;
        if (state.Mailbox is { } mailbox && mailbox.TryTake())
        {
            answer = new Answer(state, state.Version);
            return true;
        }
        answer = default;
        return false;
    }

    /// <summary>Blocks until the posted request finishes and takes its answer. For tools and tests, not frame loops.</summary>
    public Answer Take()
    {
        var state = State;
        var mailbox = state.Mailbox ?? throw new InvalidOperationException("nothing was posted to this agent");
        mailbox.WaitForResult();
        if (!mailbox.TryTake()) throw new InvalidOperationException("nothing was posted to this agent, or its answer was already taken");
        return new Answer(state, state.Version);
    }
}
