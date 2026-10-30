using System;
using System.Text;

namespace Needle.Tools;

#if NET7_0_OR_GREATER
public interface IToolSet
{
    static abstract ReadOnlySpan<byte> Schema { get; }
    static abstract ReadOnlySpan<int> GrammarNodes { get; }
    static abstract ReadOnlySpan<int> GrammarProperties { get; }
    static abstract ReadOnlySpan<int> GrammarLiterals { get; }
    static abstract ReadOnlySpan<int> GrammarUnions { get; }
    static abstract ReadOnlySpan<int> GrammarTools { get; }
    static abstract ReadOnlySpan<byte> GrammarBytes { get; }
    static abstract int GrammarRoot { get; }
    static abstract int GrammarToolCount { get; }
}

public static unsafe class ToolSession
{
    public static NeedleSession Open<TTools>(Model model, NeedleOptions? options = null) where TTools : IToolSet
    {
        var program = GrammarLoader.Load(
            TTools.GrammarNodes, TTools.GrammarProperties, TTools.GrammarLiterals,
            TTools.GrammarUnions, TTools.GrammarTools, TTools.GrammarBytes,
            TTools.GrammarRoot, TTools.GrammarToolCount);
        return new NeedleSession(model, Encoding.UTF8.GetString(TTools.Schema), program, options);
    }
}
#endif
