using System;
using System.IO;

namespace Needle;

internal sealed class SpeechModelState
{
    public readonly ModelState Model;

    public SpeechModelState(ModelState model) => Model = model;
}

/// <summary>
/// A loaded Whistle speech model. Weights are memory-mapped and shared by every transcriber made from it;
/// disposing the model only frees the weights once the last of those transcribers is disposed too.
/// </summary>
public readonly struct SpeechModel : IDisposable, IEquatable<SpeechModel>
{
    readonly int slot;
    readonly int generation;

    SpeechModel(int slot, int generation)
    {
        this.slot = slot;
        this.generation = generation;
    }

    public static SpeechModel Open(string path) => Adopt(Model.Open(path));

    public static SpeechModel Open(string path, bool prefault) => Adopt(Model.Open(path, prefault));

    /// <summary>Uses weights that already sit in memory, such as a Unity <c>NativeArray</c>. The memory must outlive the model.</summary>
    public static unsafe SpeechModel FromMemory(byte* data, nuint length) => Adopt(Model.FromMemory(data, length));

    public static SpeechModel FromBytes(byte[] bytes) => Adopt(Model.FromBytes(bytes));

    public static SpeechModel FromStream(Stream stream) => Adopt(Model.FromStream(stream));

    public static bool TryOpen(string path, out SpeechModel model, out NeedleError error)
    {
        model = default;
        error = NeedleError.None;
        try
        {
            model = Open(path);
            return true;
        }
        catch (Exception exception)
        {
            error = NeedleModel.Classify(exception);
            return false;
        }
    }

    static unsafe SpeechModel Adopt(Model model)
    {
        if (model.Audio == null)
        {
            model.Dispose();
            throw new InvalidDataException("this .cact has no speech model; open a Whistle model such as whistle.cact");
        }
        HandleTable<SpeechModelState>.Add(new SpeechModelState(new ModelState(model)), out var slot, out var generation);
        return new SpeechModel(slot, generation);
    }

    public bool IsOpen => HandleTable<SpeechModelState>.TryGet(slot, generation, out _);

    internal SpeechModelState State => HandleTable<SpeechModelState>.Get(slot, generation, nameof(SpeechModel));

    /// <summary>The engine-level model, for APIs that still take one.</summary>
    public Model Engine => State.Model.Model;

    /// <summary>Makes a transcriber. Each one owns its decoder state and serves one thread at a time.</summary>
    public Transcriber Transcriber(in TranscriberOptions options = default) => Needle.Transcriber.Create(State.Model, options);

    public bool TryTranscriber(in TranscriberOptions options, out Transcriber transcriber, out NeedleError error)
    {
        transcriber = default;
        error = NeedleError.None;
        if (!HandleTable<SpeechModelState>.TryGet(slot, generation, out var state))
        {
            error = NeedleError.Disposed;
            return false;
        }
        try
        {
            transcriber = Needle.Transcriber.Create(state.Model, options);
            return true;
        }
        catch (Exception exception)
        {
            error = NeedleModel.Classify(exception);
            return false;
        }
    }

    public void Dispose() => HandleTable<SpeechModelState>.Remove(slot, generation)?.Model.Release();

    public bool Equals(SpeechModel other) => slot == other.slot && generation == other.generation;

    public override bool Equals(object? obj) => obj is SpeechModel other && Equals(other);

    public override int GetHashCode() => (slot * 397) ^ generation;

    public static bool operator ==(SpeechModel left, SpeechModel right) => left.Equals(right);

    public static bool operator !=(SpeechModel left, SpeechModel right) => !left.Equals(right);
}

/// <summary>
/// How a transcriber runs. Every member's <c>default</c> means what upstream Whistle does, so
/// <c>default(TranscriberOptions)</c> is the right choice for most apps. The struct holds no references.
/// </summary>
public struct TranscriberOptions
{
    /// <summary>Upstream Whistle beam width.</summary>
    public const int UpstreamBeamWidth = 3;

    public static TranscriberOptions Default => default;

    /// <summary>Worker threads. 0 picks the physical performance cores, up to 4.</summary>
    public int Threads { get; set; }

    /// <summary>Full thread policy (priority, placement, spin). Overrides <see cref="Threads"/> when set.</summary>
    public ThreadPolicy? Threading { get; set; }

    /// <summary>Beam width. 0 means the upstream width of 3; 1 decodes greedily, which is faster and a little less accurate.</summary>
    public int BeamWidth { get; set; }

    /// <summary>Word timings and word confidence. Off by default, which skips the alignment pass.</summary>
    public Toggle Words { get; set; }

    /// <summary>Longest audio, in seconds, that <see cref="Transcriber.Transcribe(ReadOnlySpan{float}, int, ReadOnlySpan{byte})"/> resamples. 0 means 120.</summary>
    public int MaxSeconds { get; set; }

    /// <summary>Room for the transcript text, in UTF-8 bytes. 0 means 16 KB.</summary>
    public int MaxTextBytes { get; set; }

    internal readonly ThreadPolicy Policy => Threading ?? ThreadPolicy.Default.WithThreads(Threads > 0 ? Threads : ThreadPolicy.AutoThreads);

    internal readonly int Beam => BeamWidth > 0 ? BeamWidth : UpstreamBeamWidth;

    internal readonly int Seconds => MaxSeconds > 0 ? MaxSeconds : 120;

    internal readonly int TextBytes => MaxTextBytes > 0 ? MaxTextBytes : 16 * 1024;
}
