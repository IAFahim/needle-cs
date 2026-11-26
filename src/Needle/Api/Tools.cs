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
