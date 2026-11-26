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
                context.AddSource(model.HintName + "Tools.g.cs", Emitter.Emit(model, built.Right, triple.Right));
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
    string? DefaultLiteral)
{
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

internal sealed record ToolSetModel(string InterfaceName, string InterfaceType, string HintName, string CommandName, string CommandsName, string Namespace, bool GlobalNamespace, ImmutableArray<ToolEntry> Tools);

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
    public static BuiltToolSet Build(INamedTypeSymbol symbol)
    {
        var diagnostics = ImmutableArray.CreateBuilder<DiagnosticInfo>();
        if (symbol.TypeKind != TypeKind.Interface || symbol.Arity != 0)
        {
            diagnostics.Add(Info("NDG002", symbol.Locations.FirstOrDefault(), "The [NeedleTools] attribute applies to non-generic interfaces only."));
            return new BuiltToolSet(null, diagnostics.ToImmutable());
        }
        var tools = ImmutableArray.CreateBuilder<ToolEntry>();
        var seenAliases = new System.Collections.Generic.HashSet<string>(StringComparer.Ordinal);
        foreach (var member in symbol.GetMembers())
        {
            if (member is not IMethodSymbol method || method.MethodKind != MethodKind.Ordinary) continue;
            var tool = method.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == "Needle.Tools.ToolAttribute");
            if (tool == null) continue;
            if (method.Arity != 0)
            {
                diagnostics.Add(Info("NDG007", method.Locations.FirstOrDefault(), $"Tool '{method.Name}' must not be generic."));
                continue;
            }
            if (method.Name is "None" or "Kind" or "Presence")
            {
                diagnostics.Add(Info("NDG009", method.Locations.FirstOrDefault(), $"Tool name '{method.Name}' is reserved by the generated command types."));
                continue;
            }
            var alias = SnakeCase(method.Name);
            if (!seenAliases.Add(alias))
            {
                diagnostics.Add(Info("NDG006", method.Locations.FirstOrDefault(), $"Tool name '{method.Name}' is a duplicate after snake_casing."));
                continue;
            }
            var description = tool.ConstructorArguments.FirstOrDefault().Value as string ?? "";
            var args = ImmutableArray.CreateBuilder<ToolArg>();
            var valid = true;
            foreach (var parameter in method.Parameters)
            {
                if (BuildArg(parameter, diagnostics) is not { } arg) { valid = false; continue; }
                args.Add(arg);
            }
            if (!valid) continue;
            if (args.Select(a => a.Alias).Distinct(StringComparer.Ordinal).Count() != args.Count ||
                args.Select(a => char.ToUpperInvariant(a.Name[0]) + a.Name.Substring(1)).Distinct(StringComparer.Ordinal).Count() != args.Count)
            {
                diagnostics.Add(Info("NDG009", method.Locations.FirstOrDefault(), $"Tool '{method.Name}' has parameters whose names collide after snake_casing or PascalCasing."));
                continue;
            }
            if (args.Count > 32)
            {
                diagnostics.Add(Info("NDG003", method.Locations.FirstOrDefault(), $"Tool '{method.Name}' has more than 32 parameters."));
                continue;
            }
            tools.Add(new ToolEntry(Identifier(method.Name), alias, description, args.ToImmutable()));
        }
        var name = symbol.Name;
        var stem = name.Length > 1 && name[0] == 'I' && char.IsUpper(name[1]) ? name.Substring(1) : name;
        var commandName = stem.EndsWith("Commands", StringComparison.Ordinal) ? stem.Substring(0, stem.Length - "Commands".Length) + "Command" : stem + "Command";
        var commandsName = commandName.EndsWith("Command", StringComparison.Ordinal) ? commandName.Substring(0, commandName.Length - "Command".Length) + "Commands" : commandName + "Commands";
        var containing = symbol.ContainingNamespace;
        var globalNamespace = containing is null or { IsGlobalNamespace: true };
        var hint = new StringBuilder();
        foreach (var c in symbol.ToDisplayString()) hint.Append(char.IsLetterOrDigit(c) || c == '_' ? c : '.');
        var model = new ToolSetModel(
            name,
            symbol.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat),
            hint.ToString(),
            commandName,
            commandsName,
            globalNamespace ? "" : containing!.ToDisplayString(),
            globalNamespace,
            tools.ToImmutable());
        return new BuiltToolSet(model, diagnostics.ToImmutable());
    }

    static ToolArg? BuildArg(IParameterSymbol parameter, ImmutableArray<DiagnosticInfo>.Builder diagnostics)
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
        parameter.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == "Needle.Tools.ParamAttribute")?.ConstructorArguments.FirstOrDefault().Value as string;

    static string? DefaultLiteral(IParameterSymbol parameter)
    {
        var value = parameter.ExplicitDefaultValue;
        var type = parameter.Type;
        return type.SpecialType switch
        {
            SpecialType.System_Int32 => value is int i ? i.ToString(CultureInfo.InvariantCulture) : null,
            SpecialType.System_Int64 => value is long l ? l.ToString(CultureInfo.InvariantCulture) + "L" : null,
            SpecialType.System_Single => value is float f ? FloatLiteral(f, "float", "f") : null,
            SpecialType.System_Double => value is double d ? FloatLiteral(d, "double", "d") : null,
            SpecialType.System_Boolean => value is bool b ? (b ? "true" : "false") : null,
            _ => type.TypeKind == TypeKind.Enum && value is not null ? EnumLiteral(type, value) : null,
        };
    }

    static string FloatLiteral(double value, string type, string suffix) =>
        double.IsNaN(value) ? type + ".NaN" :
        double.IsPositiveInfinity(value) ? type + ".PositiveInfinity" :
        double.IsNegativeInfinity(value) ? type + ".NegativeInfinity" :
        (suffix == "f" ? ((float)value).ToString("R", CultureInfo.InvariantCulture) : value.ToString("R", CultureInfo.InvariantCulture)) + suffix;

    static string Identifier(string name) =>
        SyntaxFacts.GetKeywordKind(name) != SyntaxKind.None || SyntaxFacts.GetContextualKeywordKind(name) != SyntaxKind.None ? "@" + name : name;

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
            "NDG009" => "Reserved or colliding name",
            _ => "Duplicate enum literal",
        },
        id is "NDG001" or "NDG002" or "NDG003" or "NDG004" or "NDG005" or "NDG009" ? "Error" : "Warning",
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
