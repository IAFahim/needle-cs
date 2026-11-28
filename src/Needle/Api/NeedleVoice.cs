using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace Needle;

/// <summary>What <see cref="NeedleVoice.Step(FrameBudget, out Hearing)"/> heard.</summary>
public enum HearingKind : byte
{
    /// <summary>Speech began (voice activity, or <see cref="NeedleVoice.Hold"/>).</summary>
    SpeechStarted,
    /// <summary>The words so far. <see cref="Hearing.Committed"/> stays; the rest may still change.</summary>
    Partial,
    /// <summary>The phrase is over and the agent has closed a call. Provisional: the final answer may still change or drop it.</summary>
    CallReady,
    /// <summary>The voice is not sure what was meant: nothing was heard, the agent refused, or a word fell below <see cref="ListenOptions.UnsureBelow"/>.</summary>
    Unsure,
    /// <summary>The agent has answered the phrase. <see cref="Hearing.Answer"/> is the same as <see cref="NeedleAgent.Ask(ReadOnlySpan{byte})"/> of <see cref="Hearing.Transcript"/>.</summary>
    Done,
}

/// <summary>How an agent listens. Every default is sensible: zero means "use the default".</summary>
public struct ListenOptions
{
    /// <summary>Threads for speech; 0 uses the agent's.</summary>
    public int Threads { get; set; }

    /// <summary>The spoken language (such as "en"); null detects it once and keeps it.</summary>
    public string? Language { get; set; }

    /// <summary>Re-read the words in progress after this much new audio. 0.5 s by default.</summary>
    public double PartialSeconds { get; set; }

    /// <summary>Silence that ends a phrase. 0.7 s by default; ignored while held (push-to-talk).</summary>
    public double EndpointSeconds { get; set; }

    /// <summary>The longest stretch of speech before its older words are committed anyway. 8 s by default.</summary>
    public double MaxWindowSeconds { get; set; }

    /// <summary>Report <see cref="HearingKind.Unsure"/> when any word's confidence is below this. 0 (the default) turns it off.</summary>
    public float UnsureBelow { get; set; }

    /// <summary>Seconds of microphone audio the ring between <see cref="NeedleVoice.Write"/> and <see cref="NeedleVoice.Step(FrameBudget, out Hearing)"/> holds. 10 s by default.</summary>
    public double RingSeconds { get; set; }

    /// <summary>Push-to-talk only: pauses never end a phrase, only <see cref="NeedleVoice.Release"/> does.</summary>
    public bool PushToTalk { get; set; }
}

/// <summary>Where <see cref="AudioRing.Drain"/> hands samples.</summary>
internal interface ISampleSink
{
    bool Write(ReadOnlySpan<float> samples);
}

/// <summary>A single-producer, single-consumer ring of samples between an audio thread and the game thread.</summary>
internal sealed class AudioRing
{
    readonly float[] buffer;
    long written;
    long read;
    long dropped;

    public AudioRing(int capacity) => buffer = new float[Math.Max(1, capacity)];

    public long Dropped => Interlocked.Read(ref dropped);

    public int Available => (int)(Volatile.Read(ref written) - read);

    /// <summary>Producer side. Writes what fits and drops the rest; returns how many samples it took.</summary>
    public int Write(ReadOnlySpan<float> samples)
    {
        var head = written;
        var room = buffer.Length - (int)(head - Volatile.Read(ref read));
        var take = Math.Min(room, samples.Length);
        for (var i = 0; i < take; )
        {
            var at = (int)((head + i) % buffer.Length);
            var run = Math.Min(take - i, buffer.Length - at);
            samples.Slice(i, run).CopyTo(buffer.AsSpan(at, run));
            i += run;
        }
        Volatile.Write(ref written, head + take);
        if (take < samples.Length) Interlocked.Add(ref dropped, samples.Length - take);
        return take;
    }

    /// <summary>Consumer side: hands up to <paramref name="max"/> samples to <paramref name="sink"/>, in at most two runs.</summary>
    public int Drain<TSink>(TSink sink, int max) where TSink : ISampleSink
    {
        var available = (int)Math.Min(max, Volatile.Read(ref written) - read);
        var done = 0;
        while (done < available)
        {
            var at = (int)((read + done) % buffer.Length);
            var run = Math.Min(available - done, buffer.Length - at);
            sink.Write(new ReadOnlySpan<float>(buffer, at, run));
            done += run;
        }
        Volatile.Write(ref read, read + done);
        return done;
    }
}

internal enum VoicePhase : byte { Listening, Thinking }

internal struct HearingRecord
{
    public HearingKind Kind;
    public int TurnRecord;
}

internal sealed class VoiceState
{
    public readonly NeedleAgent Agent;
    public readonly AgentState AgentState;
    public readonly ModelState Speech;
    public readonly Whistle Whistle;
    public readonly SpeechStream Stream;
    public readonly AudioRing Ring;
    public readonly ListenOptions Options;
    readonly List<string> keywords;
    readonly byte[] committedText;
    readonly WordSlice[] committedWords;
    readonly byte[] pendingText;
    readonly WordSlice[] pendingWords;
    public readonly byte[] Heard;
    public readonly byte[] Shown;
    public int HeardLength;
    public int ShownLength;
    public int CommittedShown;
    public bool HeardTruncated;
    public double Lowest = 1;
    public double LowestShown = 1;
    public int HeardWords;
    public bool Started;
    public VoicePhase Phase;
    public NeedleTurn Turn;
    public int TurnVersion;
    public bool HasAnswer;
    readonly HearingRecord[] queue;
    int queueHead;
    int queueCount;
    public int Stamp;
    public bool Disposed;

    public VoiceState(NeedleAgent agent, AgentState agentState, ModelState speech, in ListenOptions options)
    {
        Agent = agent;
        AgentState = agentState;
        Speech = speech;
        Options = options;
        var session = agentState.Session;
        var policy = options.Threads > 0 ? session.Threading.WithThreads(options.Threads) : session.Threading;
        Whistle = new Whistle(speech.Model, policy) { BeamWidth = 1 };
        try
        {
            keywords = VoiceSession.EnumValues(session.ToolSet);
            Stream = new SpeechStream(Whistle, options.Language, keywords, Whistle.MaxSeconds);
            if (options.PartialSeconds > 0) Stream.PartialSeconds = options.PartialSeconds;
            Stream.EndpointSeconds = options.PushToTalk ? 0 : options.EndpointSeconds > 0 ? options.EndpointSeconds : 0.7;
            if (options.MaxWindowSeconds > 0) Stream.MaxWindowSeconds = Math.Min(options.MaxWindowSeconds, Whistle.MaxSeconds - 1);
            Ring = new AudioRing((int)((options.RingSeconds > 0 ? options.RingSeconds : 10) * Wav.SampleRate));
            var words = Whistle.MaxTranscriptTokens + 1;
            committedText = new byte[Whistle.MaxTranscriptTokens * 64];
            committedWords = new WordSlice[words];
            pendingText = new byte[Whistle.MaxTranscriptTokens * 64];
            pendingWords = new WordSlice[words];
            Heard = new byte[session.MaxInputBytes];
            Shown = new byte[session.MaxInputBytes];
            queue = new HearingRecord[agentState.OpenLog().Records.Length + 4];
        }
        catch
        {
            Whistle.Dispose();
            throw;
        }
    }

    public HearingRecord Queued(int index) => queue[index];

    void Enqueue(HearingKind kind, int turnRecord = -1)
    {
        if (queueCount < queue.Length) queue[queueCount++] = new HearingRecord { Kind = kind, TurnRecord = turnRecord };
    }

    bool TryDequeue(out int index)
    {
        if (queueHead < queueCount)
        {
            index = queueHead++;
            return true;
        }
        queueHead = queueCount = 0;
        index = -1;
        return false;
    }

    public void Drain() => Ring.Drain(Stream, Stream.Room);

    /// <summary>Works until something is heard or the budget runs out. Returns the queue index of the event, or -1.</summary>
    public int Step(FrameBudget budget)
    {
        Stamp = Stamp == int.MaxValue ? 1 : Stamp + 1;
        if (TryDequeue(out var queued)) return queued;
        Drain();
        var started = Stopwatch.GetTimestamp();
        var limit = budget.MaxMillisecondsPerSlice > 0 ? (long)(budget.MaxMillisecondsPerSlice * Stopwatch.Frequency / 1000.0) : long.MaxValue;
        var units = 0;
        for (;;)
        {
            if (!Started && Stream.HasSpeech && Phase == VoicePhase.Listening)
            {
                Started = true;
                ResetHeard();
                Enqueue(HearingKind.SpeechStarted);
                return Next();
            }
            var stepStarted = Stopwatch.GetTimestamp();
            if (Phase == VoicePhase.Thinking) units += Think(budget, units, started, limit);
            else if (Stream.Running)
            {
                units++;
                if (Stream.Step()) Heard_(Stream.Finish(committedText, committedWords, pendingText, pendingWords));
            }
            else
            {
                var kind = Stream.Due(final: false);
                if (kind == SpeechPassKind.None) return -1;
                if (!Stream.Begin(kind)) Heard_(Stream.Finish(committedText, committedWords, pendingText, pendingWords));
            }
            if (queueCount > 0) return Next();
            var now = Stopwatch.GetTimestamp();
            if (budget.MaxTokensPerSlice > 0 && units >= budget.MaxTokensPerSlice) return -1;
            if (limit != long.MaxValue && now - started + (now - stepStarted) > limit) return -1;
        }
    }

    int Next() => TryDequeue(out var index) ? index : -1;

    int Think(FrameBudget budget, int units, long started, long limit)
    {
        var session = AgentState.Session;
        var work = session.TurnWork;
        var tokens = budget.MaxTokensPerSlice > 0 ? Math.Max(1, budget.MaxTokensPerSlice - units) : 0;
        var ms = limit == long.MaxValue ? 0 : Math.Max(0.001, (limit - (Stopwatch.GetTimestamp() - started)) * 1000.0 / Stopwatch.Frequency);
        var log = AgentState.Log!;
        var first = log.Polled;
        var events = Turn.Run(new FrameBudget(tokens, ms));
        for (var i = 0; i < events.Count; i++)
        {
            var e = events[i];
            if (e.Kind == TurnEventKind.CallClosed && !e.IsWithheld) Enqueue(HearingKind.CallReady, first + i);
        }
        var spent = Math.Max(1, session.TurnWork - work);
        if (Turn.IsDone)
        {
            HasAnswer = true;
            var kind = AgentState.Kind;
            var unsure = kind == AnswerKind.Refused || (Options.UnsureBelow > 0 && Lowest < Options.UnsureBelow);
            Enqueue(unsure ? HearingKind.Unsure : HearingKind.Done);
            EndUtterance();
        }
        return spent;
    }

    // Takes in a finished speech pass: committed words join the phrase; a pause or a release ends it.
    void Heard_(StreamDelta delta)
    {
        Append(Heard, ref HeardLength, committedText, committedWords, delta.CommittedWords, ref Lowest);
        HeardWords += delta.CommittedWords;
        if (!delta.Final)
        {
            // Show the phrase so far: what is committed, then what is pending.
            Heard.AsSpan(0, HeardLength).CopyTo(Shown);
            ShownLength = HeardLength;
            CommittedShown = HeardLength;
            LowestShown = Lowest;
            Append(Shown, ref ShownLength, pendingText, pendingWords, delta.PendingWords, ref LowestShown);
            if (ShownLength > 0) Enqueue(HearingKind.Partial);
            return;
        }
        Heard.AsSpan(0, HeardLength).CopyTo(Shown);
        ShownLength = CommittedShown = HeardLength;
        LowestShown = Lowest;
        if (HeardLength == 0)
        {
            HasAnswer = false;
            Enqueue(HearingKind.Unsure);
            EndUtterance();
            return;
        }
        // Begin can only fail here if the agent is busy with other work, which throws to the caller as Ask would.
        HasAnswer = false;
        Turn = Agent.Begin(new ReadOnlySpan<byte>(Heard, 0, HeardLength));
        TurnVersion = AgentState.Version;
        Phase = VoicePhase.Thinking;
    }

    void Append(byte[] destination, ref int length, byte[] text, WordSlice[] words, int count, ref double lowest)
    {
        for (var i = 0; i < count; i++)
        {
            var w = words[i];
            var need = (length > 0 ? 1 : 0) + w.Utf8Length;
            if (length + need > destination.Length)
            {
                HeardTruncated = true;
                return;
            }
            if (length > 0) destination[length++] = (byte)' ';
            new ReadOnlySpan<byte>(text, w.Utf8Offset, w.Utf8Length).CopyTo(destination.AsSpan(length));
            length += w.Utf8Length;
            if (w.Probability < lowest) lowest = w.Probability;
        }
    }

    // The phrase is over; the next speech starts a new one. The heard text stays readable through the queued events.
    void EndUtterance()
    {
        Phase = VoicePhase.Listening;
        Started = false;
    }

    /// <summary>Starts a fresh phrase buffer when the next one begins.</summary>
    public void ResetHeard()
    {
        HeardLength = ShownLength = CommittedShown = 0;
        HeardWords = 0;
        Lowest = LowestShown = 1;
        HeardTruncated = false;
    }

    public void Cancel()
    {
        if (Phase == VoicePhase.Thinking) Turn.Cancel();
        Phase = VoicePhase.Listening;
        Started = false;
        HasAnswer = false;
        Stream.Clear();
        queueHead = queueCount = 0;
        ResetHeard();
    }
}

/// <summary>
/// An agent listening to a microphone. Feed audio with <see cref="Write"/> from any one thread (an audio callback),
/// and call <see cref="Step(FrameBudget, out Hearing)"/> once a frame on the game thread: it transcribes and thinks
/// within the budget and reports what it heard. Speech goes into the same agent, so a phrase's answer is the same,
/// bit for bit, as <see cref="NeedleAgent.Ask(ReadOnlySpan{byte})"/> of its transcript. A voice is a handle; a
/// disposed or default voice throws <see cref="ObjectDisposedException"/>.
/// </summary>
public readonly struct NeedleVoice : IDisposable, IEquatable<NeedleVoice>
{
    readonly int slot;
    readonly int generation;

    internal NeedleVoice(int slot, int generation)
    {
        this.slot = slot;
        this.generation = generation;
    }

    internal VoiceState State => HandleTable<VoiceState>.Get(slot, generation, nameof(NeedleVoice));

    public bool IsOpen => HandleTable<VoiceState>.TryGet(slot, generation, out _);

    /// <summary>The agent that answers what this voice hears.</summary>
    public NeedleAgent Agent => State.Agent;

    /// <summary>
    /// Adds 16 kHz mono samples. Safe to call from one audio thread while the game thread steps. Returns how many
    /// samples fit; the rest are dropped (see <see cref="DroppedSamples"/>) when Step has not run for too long.
    /// </summary>
    public int Write(ReadOnlySpan<float> pcm16k) => State.Ring.Write(pcm16k);

    /// <summary>Samples dropped because the ring was full.</summary>
    public long DroppedSamples => State.Ring.Dropped;

    /// <summary>True while a phrase is being heard or answered.</summary>
    public bool IsBusy
    {
        get
        {
            var state = State;
            return state.Phase == VoicePhase.Thinking || state.Started || state.Stream.Running;
        }
    }

    /// <summary>Push-to-talk: everything from now on is speech, and pauses do not end the phrase.</summary>
    public void Hold()
    {
        var state = State;
        state.Drain();
        state.Stream.Hold();
    }

    /// <summary>Push-to-talk: the phrase ends here.</summary>
    public void Release()
    {
        var state = State;
        state.Drain();
        state.Stream.Release();
    }

    /// <summary>Ends the phrase at the audio written so far, as a pause would (for a recording that has no trailing silence).</summary>
    public void EndPhrase()
    {
        var state = State;
        state.Drain();
        state.Stream.EndPhrase();
    }

    /// <summary>
    /// Transcribes and thinks within <paramref name="budget"/> and reports the next thing heard. False when nothing new
    /// was heard in this slice. A slice never splits a step (an encoder layer, one decoded token, or a model step), so
    /// every call with work to do makes progress; the token budget counts speech steps and model tokens alike. Call it
    /// again in the same frame to drain events already found; those cost nothing.
    /// </summary>
    public bool Step(FrameBudget budget, out Hearing hearing)
    {
        var state = State;
        var index = state.Step(budget);
        hearing = index < 0 ? default : new Hearing(state, state.Stamp, index);
        return index >= 0;
    }

    public bool Step(TimeSpan slice, out Hearing hearing) => Step(FrameBudget.FromMilliseconds(slice.TotalMilliseconds), out hearing);

    public bool Step(out Hearing hearing) => Step(FrameBudget.Unlimited, out hearing);

    /// <summary>Drops the phrase in progress, buffered audio and any turn it started (which, as any cancelled turn, forgets the conversation).</summary>
    public void Cancel() => State.Cancel();

    public void Dispose()
    {
        var state = HandleTable<VoiceState>.Remove(slot, generation);
        if (state == null) return;
        state.Disposed = true;
        if (state.Phase == VoicePhase.Thinking && state.Agent.IsOpen) state.Turn.Cancel();
        state.Whistle.Dispose();
        state.Speech.Release();
    }

    public bool Equals(NeedleVoice other) => slot == other.slot && generation == other.generation;

    public override bool Equals(object? obj) => obj is NeedleVoice other && Equals(other);

    public override int GetHashCode() => (slot * 397) ^ generation;

    public static bool operator ==(NeedleVoice left, NeedleVoice right) => left.Equals(right);

    public static bool operator !=(NeedleVoice left, NeedleVoice right) => !left.Equals(right);
}

/// <summary>One thing a voice heard. Valid until the voice's next <see cref="NeedleVoice.Step(FrameBudget, out Hearing)"/>.</summary>
public readonly ref struct Hearing
{
    readonly VoiceState state;
    readonly int stamp;
    readonly int index;

    internal Hearing(VoiceState state, int stamp, int index)
    {
        this.state = state;
        this.stamp = stamp;
        this.index = index;
    }

    VoiceState State
    {
        get
        {
            if (state == null) throw new InvalidOperationException("this hearing is empty");
            if (state.Disposed) throw new ObjectDisposedException(nameof(NeedleVoice));
            if (state.Stamp != stamp) throw new InvalidOperationException("the voice has stepped since this hearing; read what you need before the next Step");
            return state;
        }
    }

    public HearingKind Kind => State.Queued(index).Kind;

    /// <summary>The words of the phrase: so far for <see cref="HearingKind.Partial"/>, all of them afterwards.</summary>
    public Utf8Span Transcript
    {
        get
        {
            var s = State;
            return new Utf8Span(new ReadOnlySpan<byte>(s.Shown, 0, s.ShownLength));
        }
    }

    /// <summary>The start of <see cref="Transcript"/> that will not change any more.</summary>
    public Utf8Span Committed
    {
        get
        {
            var s = State;
            return new Utf8Span(new ReadOnlySpan<byte>(s.Shown, 0, s.CommittedShown));
        }
    }

    /// <summary>The lowest confidence of any word in <see cref="Transcript"/>, 1 when it has none.</summary>
    public double LowestWordConfidence => State.LowestShown;

    /// <summary>The call that just closed, for <see cref="HearingKind.CallReady"/>.</summary>
    public TurnEvent Call
    {
        get
        {
            var s = State;
            var record = s.Queued(index);
            if (record.Kind != HearingKind.CallReady) throw new InvalidOperationException("only a CallReady hearing has a call");
            return new TurnEvent(s.AgentState, s.TurnVersion, record.TurnRecord);
        }
    }

    /// <summary>True when the agent answered the phrase (Done, or Unsure after the agent ran).</summary>
    public bool HasAnswer
    {
        get
        {
            var s = State;
            var kind = s.Queued(index).Kind;
            return s.HasAnswer && kind is HearingKind.Done or HearingKind.Unsure;
        }
    }

    /// <summary>The agent's answer to the phrase, for Done (and Unsure when <see cref="HasAnswer"/>).</summary>
    public Answer Answer
    {
        get
        {
            if (!HasAnswer) throw new InvalidOperationException("this hearing has no answer; check HasAnswer");
            return new Answer(state.AgentState, state.TurnVersion);
        }
    }

    public override string ToString() => state == null ? "" : Kind + ": " + Transcript.ToString();
}

public readonly partial struct NeedleAgent
{
    /// <summary>
    /// Makes a voice that feeds speech into this agent. <paramref name="speech"/> is a Whistle speech model; it stays
    /// loaded until the voice is disposed. Enum values in the tools are passed to speech as keywords.
    /// </summary>
    public NeedleVoice Listen(NeedleModel speech, in ListenOptions options = default)
    {
        var state = State;
        var model = speech.State;
        model.AddReference();
        VoiceState voice;
        try
        {
            voice = new VoiceState(this, state, model, options);
        }
        catch
        {
            model.Release();
            throw;
        }
        HandleTable<VoiceState>.Add(voice, out var voiceSlot, out var voiceGeneration);
        return new NeedleVoice(voiceSlot, voiceGeneration);
    }
}
