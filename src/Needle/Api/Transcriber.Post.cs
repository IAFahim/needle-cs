using System;
using System.Threading;

namespace Needle;

internal sealed partial class TranscriberState
{
    public TranscriberMailbox? Mailbox;

    public TranscriberMailbox OpenMailbox() => Mailbox ??= new TranscriberMailbox(this);

    public void StopMailbox() => Mailbox?.Stop();
}

/// <summary>
/// One transcription in flight per transcriber, run on the transcriber's own thread. The caller copies its audio in,
/// keeps rendering, and takes the transcript on a later frame. The thread sleeps on an event while idle, so a post
/// wakes it at once and it uses no CPU between requests.
/// </summary>
internal sealed class TranscriberMailbox
{
    const int Idle = 0, Queued = 1, Running = 2, Done = 3;

    readonly TranscriberState state;
    readonly Thread thread;
    readonly ManualResetEventSlim wake = new(false);
    readonly ManualResetEventSlim finished = new(true);
    int phase;
    int length;
    int language;
    double seconds;
    volatile bool stopping;
    volatile bool cancelQueued;

    public TranscriberMailbox(TranscriberState state)
    {
        this.state = state;
        thread = new Thread(Loop) { IsBackground = true, Name = "Needle transcriber" };
        thread.Start();
    }

    public bool Busy => Volatile.Read(ref phase) is Queued or Running;

    public bool TryPost(ReadOnlySpan<float> pcm, int sampleRate, ReadOnlySpan<byte> code)
    {
        if (Busy) return false;
        var audioSeconds = sampleRate > 0 ? pcm.Length / (double)sampleRate : 0;
        var index = -1;
        if (!code.IsEmpty && (index = state.FindLanguage(code)) < 0) return Refuse(TranscriptWhy.UnknownLanguage, audioSeconds);
        if (sampleRate < 1000 || sampleRate > 768000) return Refuse(TranscriptWhy.BadSampleRate, audioSeconds);
        var count = sampleRate == Wav.SampleRate ? pcm.Length : Wav.ResampledLength(pcm.Length, sampleRate, Wav.SampleRate);
        if (count > state.Resampled.Length) return Refuse(TranscriptWhy.InputTooLong, audioSeconds);
        if (sampleRate == Wav.SampleRate) pcm.CopyTo(state.Resampled);
        else Wav.Resample(pcm, sampleRate, Wav.SampleRate, state.Resampled);
        state.Invalidate();
        cancelQueued = false;
        length = count;
        language = index;
        seconds = audioSeconds;
        finished.Reset();
        Volatile.Write(ref phase, Queued);
        wake.Set();
        return true;
    }

    bool Refuse(TranscriptWhy why, double audioSeconds)
    {
        state.Invalidate();
        state.Fail(why, audioSeconds);
        Volatile.Write(ref phase, Done);
        return true;
    }

    public void Cancel() => cancelQueued = true;

    public bool TryTake()
    {
        if (Volatile.Read(ref phase) != Done) return false;
        Volatile.Write(ref phase, Idle);
        return true;
    }

    public bool WaitForResult()
    {
        var current = Volatile.Read(ref phase);
        if (current == Idle) return false;
        finished.Wait();
        return true;
    }

    public void Stop()
    {
        if (stopping) return;
        stopping = true;
        cancelQueued = true;
        state.Whistle.Cancel();
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
            if (cancelQueued)
            {
                state.Record(default, seconds, true);
            }
            else
            {
                try
                {
                    Transcriber.Run(state, new ReadOnlySpan<float>(state.Resampled, 0, length), language, seconds);
                }
                catch
                {
                    state.Invalidate();
                    state.Fail(TranscriptWhy.Failed, seconds);
                }
            }
            Volatile.Write(ref phase, Done);
            finished.Set();
            if (stopping) return;
        }
    }
}

public readonly partial struct Transcriber
{
    /// <summary>
    /// Starts transcribing 16 kHz mono audio on the transcriber's own thread and returns at once. False when a posted
    /// transcription is still running. Copies the audio, so the caller's buffer is free on return. Take the transcript
    /// with <see cref="TryTake"/> on a later frame.
    /// </summary>
    public bool TryPost(ReadOnlySpan<float> pcm16k) => State.OpenMailbox().TryPost(pcm16k, Wav.SampleRate, default);

    /// <summary>Like <see cref="TryPost(ReadOnlySpan{float})"/>, in a given language such as <c>"en"u8</c>.</summary>
    public bool TryPost(ReadOnlySpan<float> pcm16k, ReadOnlySpan<byte> language) => State.OpenMailbox().TryPost(pcm16k, Wav.SampleRate, language);

    /// <summary>Like <see cref="TryPost(ReadOnlySpan{float})"/>, for mono audio at any sample rate, such as a 48 kHz microphone. It is resampled on the calling thread.</summary>
    public bool TryPost(ReadOnlySpan<float> pcm, int sampleRate, ReadOnlySpan<byte> language = default) => State.OpenMailbox().TryPost(pcm, sampleRate, language);

    /// <summary>Like <see cref="TryPost(ReadOnlySpan{float})"/>, but throws if a posted transcription is still running.</summary>
    public void Post(ReadOnlySpan<float> pcm16k)
    {
        if (!TryPost(pcm16k)) throw Running();
    }

    public void Post(ReadOnlySpan<float> pcm16k, ReadOnlySpan<byte> language)
    {
        if (!TryPost(pcm16k, language)) throw Running();
    }

    public void Post(ReadOnlySpan<float> pcm, int sampleRate, ReadOnlySpan<byte> language = default)
    {
        if (!TryPost(pcm, sampleRate, language)) throw Running();
    }

    /// <summary>True while a posted transcription is queued or running.</summary>
    public bool IsBusy => State.Mailbox is { } mailbox && mailbox.Busy;

    /// <summary>
    /// Takes the transcript of the last post once it is ready, without blocking. The transcript stays readable until the
    /// next post or call on this transcriber.
    /// </summary>
    public bool TryTake(out Transcript transcript)
    {
        var state = State;
        if (state.Mailbox is { } mailbox && mailbox.TryTake())
        {
            transcript = new Transcript(state, state.Version);
            return true;
        }
        transcript = default;
        return false;
    }

    /// <summary>Blocks until the posted transcription finishes and takes it. For tools and tests, not frame loops.</summary>
    public Transcript Take()
    {
        var state = State;
        var mailbox = state.Mailbox;
        if (mailbox == null || !mailbox.WaitForResult() || !mailbox.TryTake()) throw new InvalidOperationException("nothing was posted to this transcriber, or its transcript was already taken");
        return new Transcript(state, state.Version);
    }

    static InvalidOperationException Running() => new("a posted transcription is still running on this transcriber; TryTake it first");
}
