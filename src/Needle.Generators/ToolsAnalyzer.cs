using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace Needle.Generators;

/// <summary>
/// The checks the generator cannot make from the declarations it binds: where <c>[Tools]</c> sits, whether the
/// interface is partial, and a blocking <c>Ask</c> inside a Unity frame callback. Binding errors (a parameter type,
/// an <c>[Arg]</c> limit) come from the generator itself, NDG001 to NDG012, so they show even with analyzers off.
/// </summary>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class ToolsAnalyzer : DiagnosticAnalyzer
{
    const string Category = "Needle.Generators";

    internal static readonly DiagnosticDescriptor NotAnInterface = new(
        "NDG002",
        "Unsupported [Tools] target",
        "'{0}' is a {1}; [{2}] applies to interfaces only. Declare the tools as 'public partial interface I{0}' with one method per tool.",
        Category,
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor NotPartial = new(
        "NDG010",
        "[Tools] interface is not partial",
        "[Tools] interface '{0}' must be partial; the generator adds the argument structs and Calls to it. Write 'partial interface {0}'.",
        Category,
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor AskInFrame = new(
        "NDG013",
        "Blocking Ask in a Unity frame callback",
        "'{0}' runs a whole model turn (tens to hundreds of milliseconds) on the main thread inside {1}.{2}, which stalls the frame. Start it with {3} and read the answer with TryTake(out answer) in a later frame.",
        Category,
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    const string ToolsAttribute = "Needle.ToolsAttribute";
    const string LegacyToolsAttribute = "Needle.Tools.NeedleToolsAttribute";
    const string AgentType = "Needle.NeedleAgent";
    const string MonoBehaviour = "UnityEngine.MonoBehaviour";

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(NotAnInterface, NotPartial, AskInFrame);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSymbolAction(AnalyzeType, SymbolKind.NamedType);
        context.RegisterOperationAction(AnalyzeInvocation, OperationKind.Invocation);
    }

    static void AnalyzeType(SymbolAnalysisContext context)
    {
        var type = (INamedTypeSymbol)context.Symbol;
        string? attribute = null;
        foreach (var data in type.GetAttributes())
        {
            var name = data.AttributeClass?.ToDisplayString();
            if (name == ToolsAttribute) { attribute = "Tools"; break; }
            if (name == LegacyToolsAttribute) { attribute = "NeedleTools"; break; }
        }
        if (attribute == null) return;
        var location = type.Locations.Length > 0 ? type.Locations[0] : Location.None;
        if (type.TypeKind != TypeKind.Interface)
        {
            context.ReportDiagnostic(Diagnostic.Create(NotAnInterface, location, type.Name, KindName(type), attribute));
            return;
        }
        // A generic interface is the generator's NDG002; the legacy [NeedleTools] output does not extend the interface.
        if (attribute == "Tools" && !ToolModelBuilder.IsPartial(type))
            context.ReportDiagnostic(Diagnostic.Create(NotPartial, location, type.Name));
    }

    static string KindName(INamedTypeSymbol type) => type.IsRecord ? (type.IsValueType ? "record struct" : "record") : type.TypeKind switch
    {
        TypeKind.Struct => "struct",
        TypeKind.Enum => "enum",
        TypeKind.Delegate => "delegate",
        _ => "class",
    };

    static void AnalyzeInvocation(OperationAnalysisContext context)
    {
        var invocation = (IInvocationOperation)context.Operation;
        var method = invocation.TargetMethod;
        if (method.Name is not ("Ask" or "Continue") || method.ContainingType?.ToDisplayString() != AgentType) return;
        if (context.ContainingSymbol is not IMethodSymbol { IsStatic: false, Parameters.Length: 0, Name: "Update" or "LateUpdate" or "FixedUpdate" } frame) return;
        // A lambda inside Update may run later on another thread (Task.Run, a callback), so only direct calls count.
        for (var parent = invocation.Parent; parent != null; parent = parent.Parent)
            if (parent is IAnonymousFunctionOperation) return;
        if (!DerivesFrom(frame.ContainingType, MonoBehaviour)) return;
        context.ReportDiagnostic(Diagnostic.Create(AskInFrame, invocation.Syntax.GetLocation(), "NeedleAgent." + method.Name, frame.ContainingType.Name, frame.Name,
            method.Name == "Ask" ? "TryPost" : "TryPostContinue"));
    }

    static bool DerivesFrom(INamedTypeSymbol? type, string baseName)
    {
        for (var t = type?.BaseType; t != null; t = t.BaseType)
            if (t.ToDisplayString() == baseName) return true;
        return false;
    }
}
