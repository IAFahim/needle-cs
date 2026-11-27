using System;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Needle.Generators;

[Generator]
public sealed class NeedleToolsGenerator : IIncrementalGenerator
{
    const string AttributeName = "Needle.Tools.NeedleToolsAttribute";
    const string ModernAttributeName = "Needle.ToolsAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var models = context.SyntaxProvider.ForAttributeWithMetadataName(
            AttributeName,
            static (node, _) => node is InterfaceDeclarationSyntax,
            static (context, _) => ToolModelBuilder.Build((INamedTypeSymbol)context.TargetSymbol));
        var hasToolSet = context.CompilationProvider.Select(static (compilation, _) =>
            compilation.GetTypeByMetadataName("Needle.Tools.IToolSet") is not null);
        var useUtf8Literals = context.ParseOptionsProvider.Select(static (options, _) =>
            options is CSharpParseOptions cs && (int)cs.LanguageVersion >= 1100);
        var modern = context.SyntaxProvider.ForAttributeWithMetadataName(
            ModernAttributeName,
            static (node, _) => node is InterfaceDeclarationSyntax,
            static (context, _) => ToolModelBuilder.Build((INamedTypeSymbol)context.TargetSymbol, modern: true));
        context.RegisterSourceOutput(modern, static (context, built) =>
        {
            foreach (var diagnostic in built.Diagnostics)
                context.ReportDiagnostic(Diagnostic.Create(diagnostic.Descriptor, Location.Create(diagnostic.FilePath, diagnostic.Span, diagnostic.LineSpan)));
            // A [Tools] interface that is not partial gets NDG010 from ToolsAnalyzer; emitting the other half would
            // only add CS0260 on top of it.
            if (built.Model is { IsPartial: true } model)
                context.AddSource(model.InterfaceName + ".Tools.g.cs", ModernEmitter.Emit(model));
        });
        var combined = models.Combine(hasToolSet).Combine(useUtf8Literals);
        context.RegisterSourceOutput(combined.Combine(context.CompilationProvider), static (context, pair) =>
        {
            var triple = pair.Left;
            var compilation = pair.Right;
            var built = triple.Left;
            foreach (var diagnostic in built.Left.Diagnostics)
                context.ReportDiagnostic(Diagnostic.Create(diagnostic.Descriptor, Location.Create(diagnostic.FilePath, diagnostic.Span, diagnostic.LineSpan)));
            if (built.Left.IsValid && built.Left.Model is { } model)
            {
                var qualified = model.GlobalNamespace ? model.CommandsName : model.Namespace + "." + model.CommandsName;
                if (compilation.GetTypeByMetadataName(qualified) is not null) return;
                context.AddSource(model.InterfaceName + "Tools.g.cs", Emitter.Emit(model, built.Right, triple.Right));
            }
        });
    }
}

internal enum ArgType : byte { Int, Long, Float, Double, Bool, Text, Enum }

internal sealed record ToolArg(
    string Name,
    string Alias,
    ArgType Type,
    string? EnumTypeName,
    ImmutableArray<string> EnumLiterals,
    ImmutableArray<string> EnumMembers,
    int TextCapacity,
    bool FixedText,
    string? Description,
    bool Optional,
    string? DefaultLiteral,
    double? Minimum = null,
    double? Maximum = null,
    int MaxLength = 0,
    bool Inferred = false)
{
    /// <summary>The longest text the grammar and schema allow, in bytes; -1 for no limit.</summary>
    public int TextLimit => Type != ArgType.Text ? -1 : FixedText ? (MaxLength > 0 ? Math.Min(MaxLength, TextCapacity) : TextCapacity) : MaxLength > 0 ? MaxLength : -1;

    public bool PassTextThrough => Type == ArgType.Text && FixedText;
    public string SlotTypeName => Type switch
    {
        ArgType.Text => FixedText ? $"global::Needle.Tools.Utf8Text{TextCapacity}" : "global::Needle.Tools.Utf8Text128",
        ArgType.Enum => EnumTypeName!,
        _ => NameOfType(Type),
    };

    internal static string NameOfType(ArgType type) => type switch
    {
        ArgType.Int => "int",
        ArgType.Long => "long",
        ArgType.Float => "float",
        ArgType.Double => "double",
        ArgType.Bool => "bool",
        _ => "",
    };
}

internal sealed record ToolEntry(string Name, string Alias, string Description, ImmutableArray<ToolArg> Args);

internal sealed record ToolSetModel(string InterfaceName, string CommandName, string CommandsName, string Namespace, bool GlobalNamespace, ImmutableArray<ToolEntry> Tools, bool IsPartial = true)
{
    public string Q(string type) => GlobalNamespace ? type : "global::" + Namespace + "." + type;
}

internal sealed record DiagnosticInfo(string Id, string Title, string Severity, string Message, string FilePath, Microsoft.CodeAnalysis.Text.TextSpan Span, Microsoft.CodeAnalysis.Text.LinePositionSpan LineSpan)
{
    public DiagnosticDescriptor Descriptor => new(
        Id,
        Title,
        Message,
        "Needle.Generators",
        Severity == "Error" ? DiagnosticSeverity.Error : DiagnosticSeverity.Warning,
        isEnabledByDefault: true);
}

internal sealed record BuiltToolSet(ToolSetModel? Model, ImmutableArray<DiagnosticInfo> Diagnostics)
{
    public bool IsValid => Model is not null;
}

internal static class ToolModelBuilder
{
    const string DescriptionAttributeName = "System.ComponentModel.DescriptionAttribute";
    internal const string ArgAttributeName = "Needle.ArgAttribute";

    public static BuiltToolSet Build(INamedTypeSymbol symbol, bool modern = false)
    {
        var diagnostics = ImmutableArray.CreateBuilder<DiagnosticInfo>();
        if (symbol.TypeKind != TypeKind.Interface || symbol.Arity != 0)
        {
            diagnostics.Add(Info("NDG002", symbol.Locations.FirstOrDefault(), modern ? "The [Tools] attribute applies to non-generic interfaces only." : "The [NeedleTools] attribute applies to non-generic interfaces only."));
            return new BuiltToolSet(null, diagnostics.ToImmutable());
        }
        var tools = ImmutableArray.CreateBuilder<ToolEntry>();
        var seenAliases = new System.Collections.Generic.HashSet<string>(StringComparer.Ordinal);
        foreach (var member in symbol.GetMembers())
        {
            if (member is not IMethodSymbol method || method.MethodKind != MethodKind.Ordinary) continue;
            var tool = method.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == "Needle.Tools.ToolAttribute");
            if (tool == null && !modern) continue;
            if (modern && method.IsStatic) continue;
            if (method.Arity != 0)
            {
                diagnostics.Add(Info("NDG007", method.Locations.FirstOrDefault(), $"Tool '{method.Name}' must not be generic."));
                continue;
            }
            var alias = SnakeCase(method.Name);
            if (!seenAliases.Add(alias))
            {
                diagnostics.Add(Info("NDG006", method.Locations.FirstOrDefault(), $"Tool name '{method.Name}' is a duplicate after snake_casing."));
                continue;
            }
            var description = tool?.ConstructorArguments.FirstOrDefault().Value as string
                ?? DescriptionOf(method)
                ?? "";
            if (modern && description.Length == 0)
                diagnostics.Add(Info("NDG009", method.Locations.FirstOrDefault(), $"Tool '{method.Name}' has no description; add [Description(\"...\")] so the model knows when to call it."));
            var args = ImmutableArray.CreateBuilder<ToolArg>();
            var valid = true;
            foreach (var parameter in method.Parameters)
            {
                if (BuildArg(parameter, diagnostics) is not { } arg) { valid = false; continue; }
                args.Add(arg);
            }
            if (!valid) continue;
            if (args.Count > 32)
            {
                diagnostics.Add(Info("NDG003", method.Locations.FirstOrDefault(), $"Tool '{method.Name}' has more than 32 parameters."));
                continue;
            }
            tools.Add(new ToolEntry(method.Name, alias, description, args.ToImmutable()));
        }
        var name = symbol.Name;
        var stem = name.StartsWith("I", StringComparison.Ordinal) && name.Length > 1 ? name.Substring(1) : name;
        var commandName = stem.EndsWith("Commands", StringComparison.Ordinal) ? stem.Substring(0, stem.Length - "Commands".Length) + "Command" : stem + "Command";
        var commandsName = commandName.EndsWith("Command", StringComparison.Ordinal) ? commandName.Substring(0, commandName.Length - "Command".Length) + "Commands" : commandName + "Commands";
        var containing = symbol.ContainingNamespace;
        var globalNamespace = containing is null or { IsGlobalNamespace: true };
        var model = new ToolSetModel(
            name,
            commandName,
            commandsName,
            globalNamespace ? "" : containing!.ToDisplayString(),
            globalNamespace,
            tools.ToImmutable(),
            IsPartial(symbol));
        return new BuiltToolSet(model, diagnostics.ToImmutable());
    }

    internal static bool IsPartial(INamedTypeSymbol symbol)
    {
        foreach (var reference in symbol.DeclaringSyntaxReferences)
            if (reference.GetSyntax() is TypeDeclarationSyntax declaration && !declaration.Modifiers.Any(SyntaxKind.PartialKeyword))
                return false;
        return true;
    }

    static ToolArg? BuildArg(IParameterSymbol parameter, ImmutableArray<DiagnosticInfo>.Builder diagnostics)
    {
        if (BuildType(parameter, diagnostics) is not { } arg) return null;
        return ApplyLimits(parameter, arg, diagnostics);
    }

    // [Arg(Min, Max, MaxLength, Inferred)]: checked here so a bad limit is an error at the parameter, never a schema
    // the model reads one way and the repair step another.
    static ToolArg? ApplyLimits(IParameterSymbol parameter, ToolArg arg, ImmutableArray<DiagnosticInfo>.Builder diagnostics)
    {
        var attribute = parameter.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == ArgAttributeName);
        if (attribute == null) return arg;
        double? min = null, max = null;
        var maxLength = 0;
        var inferred = false;
        foreach (var named in attribute.NamedArguments)
        {
            var value = named.Value.Value;
            switch (named.Key)
            {
                case "Min" when value is double d && !double.IsNaN(d): min = d; break;
                case "Max" when value is double d && !double.IsNaN(d): max = d; break;
                case "MaxLength" when value is int i: maxLength = i; break;
                case "Inferred" when value is bool b: inferred = b; break;
            }
        }
        var location = parameter.Locations.FirstOrDefault();
        var numeric = arg.Type is ArgType.Int or ArgType.Long or ArgType.Float or ArgType.Double;
        if ((min.HasValue || max.HasValue) && !numeric)
        {
            diagnostics.Add(Info("NDG012", location, $"[Arg] Min and Max apply to numeric parameters; '{parameter.Name}' is {parameter.Type.ToDisplayString()}."));
            return null;
        }
        if (min is { } low && max is { } high && low > high)
        {
            diagnostics.Add(Info("NDG012", location, $"[Arg] on '{parameter.Name}' has Min {Number(low)} above Max {Number(high)}."));
            return null;
        }
        if (min is { } m && double.IsInfinity(m) || max is { } x && double.IsInfinity(x))
        {
            diagnostics.Add(Info("NDG012", location, $"[Arg] on '{parameter.Name}' needs finite Min and Max; leave a side unset for no limit."));
            return null;
        }
        if (maxLength != 0 && (arg.Type != ArgType.Text || maxLength < 0))
        {
            diagnostics.Add(Info("NDG012", location, arg.Type != ArgType.Text
                ? $"[Arg] MaxLength applies to text parameters; '{parameter.Name}' is {parameter.Type.ToDisplayString()}."
                : $"[Arg] MaxLength on '{parameter.Name}' must be positive."));
            return null;
        }
        if (arg.FixedText && maxLength > arg.TextCapacity)
        {
            diagnostics.Add(Info("NDG011", location,
                $"Parameter '{parameter.Name}' is {parameter.Type.Name}, which holds {arg.TextCapacity} UTF-8 bytes, but [Arg(MaxLength = {maxLength})] allows {maxLength}; " +
                $"text past byte {arg.TextCapacity} would be cut. Use {LargerText(maxLength)} or lower MaxLength."));
            return null;
        }
        return arg with { Minimum = min, Maximum = max, MaxLength = maxLength, Inferred = inferred };
    }

    static string LargerText(int bytes)
    {
        foreach (var capacity in new[] { 8, 16, 32, 64, 128, 256 })
            if (capacity >= bytes) return "Utf8Text" + capacity.ToString(CultureInfo.InvariantCulture);
        return "string";
    }

    internal static string Number(double value) =>
        value == Math.Floor(value) && Math.Abs(value) < 1e15 ? ((long)value).ToString(CultureInfo.InvariantCulture) : value.ToString("R", CultureInfo.InvariantCulture);

    static ToolArg? BuildType(IParameterSymbol parameter, ImmutableArray<DiagnosticInfo>.Builder diagnostics)
    {
        if (parameter.RefKind != RefKind.None)
        {
            diagnostics.Add(Info("NDG004", parameter.Locations.FirstOrDefault(), $"Parameter '{parameter.Name}' must not be ref/out/in."));
            return null;
        }
        var type = parameter.Type;
        var optional = parameter.HasExplicitDefaultValue;
        string? defaultLiteral = null;
        if (optional)
        {
            defaultLiteral = DefaultLiteral(parameter);
            if (defaultLiteral == null)
            {
                diagnostics.Add(Info("NDG005", parameter.Locations.FirstOrDefault(), $"Parameter '{parameter.Name}' needs a constant numeric, bool or enum default."));
                return null;
            }
        }
        switch (type.SpecialType)
        {
            case SpecialType.System_Int32: return Literal(parameter, ArgType.Int, null, default, 0, false, optional, defaultLiteral);
            case SpecialType.System_Int64: return Literal(parameter, ArgType.Long, null, default, 0, false, optional, defaultLiteral);
            case SpecialType.System_Single: return Literal(parameter, ArgType.Float, null, default, 0, false, optional, defaultLiteral);
            case SpecialType.System_Double: return Literal(parameter, ArgType.Double, null, default, 0, false, optional, defaultLiteral);
            case SpecialType.System_Boolean: return Literal(parameter, ArgType.Bool, null, default, 0, false, optional, defaultLiteral);
            case SpecialType.System_String:
                if (optional)
                {
                    diagnostics.Add(Info("NDG005", parameter.Locations.FirstOrDefault(), $"Parameter '{parameter.Name}' needs a constant numeric, bool or enum default."));
                    return null;
                }
                return Literal(parameter, ArgType.Text, null, default, 128, false, optional, null);
            default: break;
        }
        if (type.TypeKind == TypeKind.Enum)
        {
            var members = type.GetMembers().OfType<IFieldSymbol>().Where(f => f.HasConstantValue).Select(f => f.Name).ToList();
            var literals = members.Select(m => SnakeCase(m)).ToList();
            if (literals.Count == 0 || literals.Count != literals.Distinct(StringComparer.Ordinal).Count())
            {
                diagnostics.Add(Info("NDG008", parameter.Locations.FirstOrDefault(), $"Enum '{type.Name}' needs distinct snake_cased member names."));
                return null;
            }
            return new ToolArg(parameter.Name, SnakeCase(parameter.Name), ArgType.Enum, type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
                literals.ToImmutableArray(), members.ToImmutableArray(), 0, false, Description(parameter), optional, defaultLiteral);
        }
        if (type is INamedTypeSymbol { Arity: 0 } named && named.Name.StartsWith("Utf8Text", StringComparison.Ordinal) &&
            int.TryParse(named.Name.Substring("Utf8Text".Length), NumberStyles.None, CultureInfo.InvariantCulture, out var capacity) &&
            capacity is 8 or 16 or 32 or 64 or 128 or 256)
            return Literal(parameter, ArgType.Text, named.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat), default(ImmutableArray<string>), capacity, true, optional, defaultLiteral);
        diagnostics.Add(Info("NDG001", parameter.Locations.FirstOrDefault(),
            $"Parameter '{parameter.Name}' has unsupported type '{type.ToDisplayString()}'; use int, long, float, double, bool, string, Utf8Text8-256 or an enum."));
        return null;

        static ToolArg Literal(IParameterSymbol parameter, ArgType type, string? enumTypeName, ImmutableArray<string> enumMembers, int capacity, bool fixedText, bool optional, string? defaultLiteral) =>
            new(parameter.Name, SnakeCase(parameter.Name), type, enumTypeName, default, enumMembers, capacity, fixedText, Description(parameter), optional, defaultLiteral);
    }

    static string? Description(IParameterSymbol parameter) =>
        parameter.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == "Needle.Tools.ParamAttribute")?.ConstructorArguments.FirstOrDefault().Value as string
        ?? DescriptionOf(parameter);

    static string? DescriptionOf(ISymbol symbol) =>
        symbol.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == DescriptionAttributeName)?.ConstructorArguments.FirstOrDefault().Value as string;

    static string? DefaultLiteral(IParameterSymbol parameter)
    {
        var value = parameter.ExplicitDefaultValue;
        var type = parameter.Type;
        return type.SpecialType switch
        {
            SpecialType.System_Int32 => value is int i ? i.ToString(CultureInfo.InvariantCulture) : null,
            SpecialType.System_Int64 => value is long l ? l.ToString(CultureInfo.InvariantCulture) + "L" : null,
            SpecialType.System_Single => value is float f ? f.ToString("R", CultureInfo.InvariantCulture) + "f" : null,
            SpecialType.System_Double => value is double d ? d.ToString("R", CultureInfo.InvariantCulture) + "d" : null,
            SpecialType.System_Boolean => value is bool b ? (b ? "true" : "false") : null,
            _ => type.TypeKind == TypeKind.Enum && value is not null ? EnumLiteral(type, value) : null,
        };
    }

    static string EnumLiteral(ITypeSymbol type, object value)
    {
        foreach (var member in type.GetMembers().OfType<IFieldSymbol>())
            if (member.HasConstantValue && Equals(member.ConstantValue, value))
                return type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) + "." + member.Name;
        return "(" + type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat) + ")" + Convert.ToInt64(value, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
    }

    static DiagnosticInfo Info(string id, Location? location, string message) => new(
        id,
        id switch
        {
            "NDG001" => "Unsupported parameter type",
            "NDG002" => "Unsupported [NeedleTools] target",
            "NDG003" => "Too many parameters",
            "NDG004" => "Unsupported parameter passing",
            "NDG005" => "Unsupported default value",
            "NDG006" => "Duplicate tool name",
            "NDG007" => "Generic tool method",
            "NDG009" => "Tool without a description",
            "NDG011" => "Text limit larger than the fixed-size text",
            "NDG012" => "Invalid [Arg] limits",
            _ => "Duplicate enum literal",
        },
        id is "NDG001" or "NDG002" or "NDG003" or "NDG004" or "NDG005" or "NDG011" or "NDG012" ? "Error" : "Warning",
        message,
        location?.SourceTree?.FilePath ?? "",
        location?.SourceSpan ?? default,
        location?.GetLineSpan().Span ?? default);

    public static string SnakeCase(string name)
    {
        var builder = new StringBuilder();
        for (var i = 0; i < name.Length; i++)
        {
            var c = name[i];
            if (char.IsUpper(c))
            {
                var boundary = i > 0 && (char.IsLower(name[i - 1]) || char.IsDigit(name[i - 1]) || (char.IsUpper(name[i - 1]) && i + 1 < name.Length && char.IsLower(name[i + 1])));
                if (boundary && builder.Length > 0 && builder[builder.Length - 1] != '_') builder.Append('_');
                builder.Append(char.ToLowerInvariant(c));
            }
            else if (char.IsLetterOrDigit(c) || c == '_') builder.Append(c);
            else if (builder.Length > 0 && builder[builder.Length - 1] != '_') builder.Append('_');
        }
        return builder.ToString().Trim('_') is { Length: > 0 } result ? result : "tool";
    }
}
