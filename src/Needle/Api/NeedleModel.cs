using System;
using System.IO;
using System.Text;
using System.Threading;

namespace Needle;

/// <summary>What went wrong when a <c>Try</c> call returned <c>false</c>.</summary>
public enum NeedleError : byte
{
    None,
    NotFound,
    AccessDenied,
    InvalidModel,
    Unsupported,
    InvalidTools,
    Disposed,
    IoFailed,
}

/// <summary>Owns the loaded weights and counts the agents that use them.</summary>
internal sealed class ModelState
{
    public readonly Model Model;
    int references = 1;

    public ModelState(Model model) => Model = model;

    public void AddReference() => Interlocked.Increment(ref references);

    public void Release()
    {
        if (Interlocked.Decrement(ref references) == 0) Model.Dispose();
    }
}

/// <summary>
/// A loaded Needle model. Weights are memory-mapped and shared by every agent made from it; disposing the
/// model only frees the weights once the last of those agents is disposed too.
/// </summary>
public readonly struct NeedleModel : IDisposable, IEquatable<NeedleModel>
{
    readonly int slot;
    readonly int generation;

    NeedleModel(int slot, int generation)
    {
        this.slot = slot;
        this.generation = generation;
    }

    public static NeedleModel Open(string path) => Adopt(Model.Open(path));

    public static NeedleModel Open(string path, bool prefault) => Adopt(Model.Open(path, prefault));

    /// <summary>Uses weights that already sit in memory, such as a Unity <c>NativeArray</c>. The memory must outlive the model.</summary>
    public static unsafe NeedleModel FromMemory(byte* data, nuint length) => Adopt(Model.FromMemory(data, length));

    public static NeedleModel FromBytes(byte[] bytes) => Adopt(Model.FromBytes(bytes));

    public static NeedleModel FromStream(Stream stream) => Adopt(Model.FromStream(stream));

    public static bool TryOpen(string path, out NeedleModel model, out NeedleError error)
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
            error = Classify(exception);
            return false;
        }
    }

    internal static NeedleError Classify(Exception exception) => exception switch
    {
        FileNotFoundException or DirectoryNotFoundException => NeedleError.NotFound,
        UnauthorizedAccessException => NeedleError.AccessDenied,
        InvalidDataException or EndOfStreamException => NeedleError.InvalidModel,
        NotSupportedException => NeedleError.Unsupported,
        ObjectDisposedException => NeedleError.Disposed,
        FormatException or ArgumentException => NeedleError.InvalidTools,
        _ => NeedleError.IoFailed,
    };

    static NeedleModel Adopt(Model model)
    {
        HandleTable<ModelState>.Add(new ModelState(model), out var slot, out var generation);
        return new NeedleModel(slot, generation);
    }

    public bool IsOpen => HandleTable<ModelState>.TryGet(slot, generation, out _);

    internal ModelState State => HandleTable<ModelState>.Get(slot, generation, nameof(NeedleModel));

    /// <summary>The engine-level model, for APIs that still take one.</summary>
    public Model Engine => State.Model;

    /// <summary>Makes an agent that calls the tools described by an OpenAI-style JSON array.</summary>
    public NeedleAgent Agent(string toolsJson, in AgentOptions options = default) => NeedleAgent.Create(State, toolsJson, options);

    /// <summary>Makes an agent for the tools of a <c>[Tools]</c> interface.</summary>
    public NeedleAgent Agent<TTools>(in AgentOptions options = default) where TTools : class =>
        NeedleAgent.Create(State, ToolCatalog<TTools>.Json, options);

    public NeedleAgent Agent(ReadOnlySpan<byte> toolsJsonUtf8, in AgentOptions options = default) =>
        NeedleAgent.Create(State, Encoding.UTF8.GetString(toolsJsonUtf8), options);

    public bool TryAgent(string toolsJson, in AgentOptions options, out NeedleAgent agent, out NeedleError error)
    {
        agent = default;
        error = NeedleError.None;
        if (!HandleTable<ModelState>.TryGet(slot, generation, out var state))
        {
            error = NeedleError.Disposed;
            return false;
        }
        try
        {
            agent = NeedleAgent.Create(state, toolsJson, options);
            return true;
        }
        catch (Exception exception)
        {
            error = Classify(exception);
            return false;
        }
    }

    public void Dispose() => HandleTable<ModelState>.Remove(slot, generation)?.Release();

    public bool Equals(NeedleModel other) => slot == other.slot && generation == other.generation;

    public override bool Equals(object? obj) => obj is NeedleModel other && Equals(other);

    public override int GetHashCode() => (slot * 397) ^ generation;

    public static bool operator ==(NeedleModel left, NeedleModel right) => left.Equals(right);

    public static bool operator !=(NeedleModel left, NeedleModel right) => !left.Equals(right);
}
