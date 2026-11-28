using System;

namespace Needle;

/// <summary>
/// Marks a <c>partial interface</c> whose methods are tools the model can call. The generator writes the JSON schema,
/// an arguments struct per method (<c>SetTimerArgs</c> for <c>SetTimer</c>) and a <c>Calls.Dispatch</c> that invokes
/// your implementation. Describe each method and parameter with <see cref="System.ComponentModel.DescriptionAttribute"/>.
/// </summary>
[AttributeUsage(AttributeTargets.Interface, AllowMultiple = false, Inherited = false)]
public sealed class ToolsAttribute : Attribute
{
}

/// <summary>
/// Limits and grounding for one parameter of a <c>[Tools]</c> method. The generator writes <see cref="Min"/>,
/// <see cref="Max"/> and <see cref="MaxLength"/> into the JSON schema (<c>minimum</c>, <c>maximum</c>, <c>maxLength</c>),
/// so the model sees them, and the repair step checks every call against them: a call whose number falls outside
/// <c>[Min, Max]</c> is withheld, not clamped, and the answer carries <see cref="ReplyFlags.Invalid"/>
/// (<see cref="AnswerWhy.OutOfRange"/> when no call is left). Named <c>Arg</c> so it does not clash with
/// <c>UnityEngine.RangeAttribute</c> or <c>UnityEngine.MinAttribute</c>.
/// </summary>
[AttributeUsage(AttributeTargets.Parameter, AllowMultiple = false, Inherited = false)]
public sealed class ArgAttribute : Attribute
{
    /// <summary>The smallest number allowed, inclusive. Numeric parameters only; unset (NaN) means no lower limit.</summary>
    public double Min { get; set; } = double.NaN;

    /// <summary>The largest number allowed, inclusive. Numeric parameters only; unset (NaN) means no upper limit.</summary>
    public double Max { get; set; } = double.NaN;

    /// <summary>
    /// The longest text allowed, in characters: the schema's <c>maxLength</c>, which the grammar enforces. Text parameters
    /// only; 0 means the type's own limit (the capacity of a <c>Utf8TextN</c>, none for <c>string</c>). A <c>Utf8TextN</c>
    /// holds N UTF-8 bytes, so it may not exceed N, and text with multi-byte characters can still fill the bytes first.
    /// </summary>
    public int MaxLength { get; set; }

    /// <summary>
    /// The value is one the game works out rather than one the player says (coordinates, a target id). An inferred
    /// field is exempt from the grounding gate and from the repair rules that drop or replace values the request never
    /// states. Range limits still apply.
    /// </summary>
    public bool Inferred { get; set; }
}

/// <summary>Carries the generated JSON schema of a <c>[Tools]</c> interface. Written by the generator; do not apply by hand.</summary>
[AttributeUsage(AttributeTargets.Interface, AllowMultiple = false, Inherited = false)]
public sealed class ToolSchemaAttribute : Attribute
{
    public ToolSchemaAttribute(string json) => Json = json;

    public string Json { get; }
}

/// <summary>The arguments of one tool, read from a <see cref="Call"/>. Implemented by generated structs.</summary>
public interface IToolArgs<TSelf> where TSelf : struct, IToolArgs<TSelf>
{
    /// <summary>The tool's name in the schema.</summary>
    string ToolName { get; }

    /// <summary>Reads the arguments of a call to this tool. False when a required argument is missing or malformed.</summary>
    bool TryRead(Call call, out TSelf args);
}

internal static class ToolCatalog<TTools> where TTools : class
{
    public static readonly string Json = Find();

    static string Find()
    {
        var type = typeof(TTools);
        foreach (var attribute in type.GetCustomAttributes(typeof(ToolSchemaAttribute), false))
            return ((ToolSchemaAttribute)attribute).Json;
        throw new InvalidOperationException($"{type.Name} has no generated tool schema; declare it as a partial interface marked [Tools] and reference the Needle generator");
    }
}
