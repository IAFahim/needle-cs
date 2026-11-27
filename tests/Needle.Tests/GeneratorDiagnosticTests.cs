using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Needle.Generators;
using Xunit;

namespace Needle.Tests;

// Drives the generator and ToolsAnalyzer through Roslyn on small C# 9 sources (Unity's language version), with
// stand-ins for the UnityEngine types, and checks each diagnostic's ID, message and place.
public sealed class GeneratorDiagnosticTests
{
    const string Usings = "using System.ComponentModel;\nusing Needle;\nusing Needle.Tools;\n";

    const string UnityStubs = """
        namespace UnityEngine
        {
            public class Object { }
            public class Component : Object { }
            public class Behaviour : Component { }
            public class MonoBehaviour : Behaviour { }
        }
        """;

    static readonly Lazy<MetadataReference[]> References = new(() =>
    {
        var paths = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator).ToList();
        paths.Add(typeof(NeedleAgent).Assembly.Location);
        return paths.Distinct().Select(p => (MetadataReference)MetadataReference.CreateFromFile(p)).ToArray();
    });

    sealed record Result(ImmutableArray<Diagnostic> Generator, ImmutableArray<Diagnostic> Analyzer, ImmutableArray<Diagnostic> Compiler, string Generated, string Source)
    {
        // Generator diagnostics carry a file location with no syntax tree, so the span is read from the source itself.
        public string Marked(Diagnostic diagnostic)
        {
            var text = diagnostic.Location.SourceTree?.ToString() ?? Source;
            Assert.Equal("Probe.cs", diagnostic.Location.GetLineSpan().Path);
            return text.Substring(diagnostic.Location.SourceSpan.Start, diagnostic.Location.SourceSpan.Length);
        }

        public IEnumerable<Diagnostic> Needle => Generator.Concat(Analyzer).Where(d => d.Id.StartsWith("NDG", StringComparison.Ordinal));
        public IEnumerable<Diagnostic> Errors => Compiler.Where(d => d.Severity == DiagnosticSeverity.Error);
    }

    Result? last;

    // xUnit makes a new instance per test, so the last run is this test's own.
    string Marked(Diagnostic diagnostic) => last!.Marked(diagnostic);

    Result Run(string source)
    {
        var parse = new CSharpParseOptions(LanguageVersion.CSharp9);
        var compilation = CSharpCompilation.Create("Probe",
            new[] { CSharpSyntaxTree.ParseText(Usings + source, parse, "Probe.cs"), CSharpSyntaxTree.ParseText(UnityStubs, parse, "Unity.cs") },
            References.Value,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true, nullableContextOptions: NullableContextOptions.Enable));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new[] { new NeedleToolsGenerator().AsSourceGenerator() }, parseOptions: parse);
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var output, out var generatorDiagnostics);
        var analyzer = output.WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(new ToolsAnalyzer())).GetAnalyzerDiagnosticsAsync().GetAwaiter().GetResult();
        var generated = string.Join("\n", driver.GetRunResult().GeneratedTrees.Select(t => t.ToString()));
        return last = new Result(generatorDiagnostics, analyzer, output.GetDiagnostics(), generated, Usings + source);
    }

    static Diagnostic Single(Result result, string id)
    {
        var found = result.Needle.Where(d => d.Id == id).ToList();
        Assert.True(found.Count == 1, $"expected one {id}, got: {string.Join("; ", result.Needle.Select(d => d.Id + " " + d.GetMessage()))}");
        return found[0];
    }

    [Fact]
    public void ValidToolsCompileAsCSharp9WithNoDiagnostics()
    {
        var result = Run("""
            [Tools] public partial interface IKitchen
            {
                [Description("Set a countdown timer.")] void SetTimer([Description("Length in minutes"), Arg(Min = 1, Max = 99)] int minutes);
                [Description("Rename the timer.")] void Name([Arg(MaxLength = 32)] Utf8Text32 label);
                [Description("Walk.")] void Walk([Arg(Inferred = true)] float x, [Arg(Inferred = true)] float y);
            }
            """);
        Assert.Empty(result.Needle);
        Assert.Empty(result.Errors);
        Assert.Contains("\\\"minimum\\\":1,\\\"maximum\\\":99", result.Generated);
        Assert.Contains("\\\"inferred\\\":[\\\"x\\\",\\\"y\\\"]", result.Generated);
    }

    [Fact]
    public void NotPartialIsNdg010AndNothingIsEmitted()
    {
        var result = Run("""
            [Tools] public interface IKitchen
            {
                [Description("Set a countdown timer.")] void SetTimer(int minutes);
            }
            """);
        var diagnostic = Single(result, "NDG010");
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal("IKitchen", Marked(diagnostic));
        Assert.Contains("must be partial", diagnostic.GetMessage());
        Assert.Equal("", result.Generated);
        Assert.DoesNotContain(result.Compiler, d => d.Id == "CS0260");
    }

    [Fact]
    public void LegacyNeedleToolsNeedNotBePartial()
    {
        var result = Run("""
            [NeedleTools] public interface ILegacyCommands
            {
                [Tool("Set a countdown timer.")] void SetTimer(int minutes);
            }
            """);
        // Its output implements the net10 IToolSet (static abstract members), which C# 9 cannot compile; Unity builds
        // against the netstandard2.1 plugin, where that interface does not exist. Only the diagnostics matter here.
        Assert.Empty(result.Needle);
    }

    [Theory]
    [InlineData("public partial class Kitchen { }", "class")]
    [InlineData("public partial struct Kitchen { }", "struct")]
    [InlineData("public partial record Kitchen;", "record")]
    public void ToolsOnANonInterfaceIsNdg002(string declaration, string kind)
    {
        var diagnostic = Single(Run("[Tools] " + declaration), "NDG002");
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal("Kitchen", Marked(diagnostic));
        Assert.Contains($"'Kitchen' is a {kind}; [Tools] applies to interfaces only", diagnostic.GetMessage());
    }

    [Fact]
    public void GenericToolsInterfaceIsNdg002()
    {
        var diagnostic = Single(Run("[Tools] public partial interface IKitchen<T> { void SetTimer(int minutes); }"), "NDG002");
        Assert.Contains("non-generic interfaces", diagnostic.GetMessage());
    }

    [Fact]
    public void UnbindableParameterTypeIsNdg001()
    {
        var diagnostic = Single(Run("""
            [Tools] public partial interface IKitchen
            {
                [Description("Set an alarm.")] void SetAlarm(System.DateTime when);
            }
            """), "NDG001");
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal("when", Marked(diagnostic));
        Assert.Contains("unsupported type 'System.DateTime'", diagnostic.GetMessage());
    }

    [Fact]
    public void RefParameterIsNdg004AndGenericMethodIsNdg007()
    {
        var result = Run("""
            [Tools] public partial interface IKitchen
            {
                [Description("Set a timer.")] void SetTimer(ref int minutes);
                [Description("Store a thing.")] void Store<T>(int count);
            }
            """);
        Assert.Contains("must not be ref/out/in", Single(result, "NDG004").GetMessage());
        Assert.Contains("must not be generic", Single(result, "NDG007").GetMessage());
    }

    [Fact]
    public void TextLimitPastTheFixedSizeIsNdg011()
    {
        var diagnostic = Single(Run("""
            [Tools] public partial interface ISquad
            {
                [Description("Give an item.")] void Give([Arg(MaxLength = 40)] Utf8Text32 member);
            }
            """), "NDG011");
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Equal("member", Marked(diagnostic));
        Assert.Equal("Parameter 'member' is Utf8Text32, which holds 32 UTF-8 bytes, but [Arg(MaxLength = 40)] allows 40 characters; text past byte 32 would be cut. Use Utf8Text64 or lower MaxLength.",
            diagnostic.GetMessage());
    }

    [Fact]
    public void TextLimitWithinTheFixedSizeNarrowsTheSchema()
    {
        var result = Run("""
            [Tools] public partial interface ISquad
            {
                [Description("Give an item.")] void Give([Arg(MaxLength = 12)] Utf8Text32 member, [Arg(MaxLength = 300)] string note);
            }
            """);
        Assert.Empty(result.Needle);
        Assert.Contains("\\\"member\\\":{\\\"type\\\":\\\"string\\\",\\\"maxLength\\\":12}", result.Generated);
        Assert.Contains("\\\"note\\\":{\\\"type\\\":\\\"string\\\",\\\"maxLength\\\":300}", result.Generated);
    }

    [Theory]
    [InlineData("[Arg(Min = 1)] string name", "Min and Max apply to numeric parameters")]
    [InlineData("[Arg(Min = 10, Max = 1)] int count", "has Min 10 above Max 1")]
    [InlineData("[Arg(MaxLength = 8)] int count", "MaxLength applies to text parameters")]
    [InlineData("[Arg(MaxLength = -1)] string name", "must be positive")]
    [InlineData("[Arg(Max = double.PositiveInfinity)] double scale", "needs finite Min and Max")]
    public void BadArgLimitsAreNdg012(string parameter, string message)
    {
        var diagnostic = Single(Run("[Tools] public partial interface IKitchen { [Description(\"Do it.\")] void Do(" + parameter + "); }"), "NDG012");
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains(message, diagnostic.GetMessage());
    }

    const string Agent = """
        [Tools] public partial interface IKitchen { [Description("Set a countdown timer.")] void SetTimer(int minutes); }
        """;

    [Theory]
    [InlineData("Update")]
    [InlineData("LateUpdate")]
    [InlineData("FixedUpdate")]
    public void AskInAFrameCallbackIsNdg013(string callback)
    {
        var diagnostic = Single(Run(Agent + """
            public sealed class Chef : UnityEngine.MonoBehaviour
            {
                NeedleAgent agent;
                void
            """ + " " + callback + """
            () { var answer = agent.Ask("set a timer for 5 minutes"); }
            }
            """), "NDG013");
        Assert.Equal(DiagnosticSeverity.Warning, diagnostic.Severity);
        Assert.Equal("agent.Ask(\"set a timer for 5 minutes\")", Marked(diagnostic));
        Assert.Equal($"'NeedleAgent.Ask' runs a whole model turn (tens to hundreds of milliseconds) on the main thread inside Chef.{callback}, which stalls the frame. " +
            "Start it with TryPost and read the answer with TryTake(out answer) in a later frame.", diagnostic.GetMessage());
    }

    [Fact]
    public void ContinueInUpdateSuggestsTryPostContinue()
    {
        var diagnostic = Single(Run(Agent + """
            public class Chef : UnityEngine.MonoBehaviour
            {
                NeedleAgent agent;
                void Update() { agent.Continue("{\"ok\":true}"); }
            }
            public sealed class SousChef : Chef { }
            """), "NDG013");
        Assert.Contains("Start it with TryPostContinue", diagnostic.GetMessage());
    }

    [Fact]
    public void AskOutsideAFrameCallbackIsFine()
    {
        var result = Run(Agent + """
            public sealed class Chef : UnityEngine.MonoBehaviour
            {
                NeedleAgent agent;
                void Start() { agent.Ask("warm up"); }
                void Update()
                {
                    agent.TryPost("set a timer for 5 minutes");
                    if (agent.TryTake(out var answer)) { }
                    System.Threading.Tasks.Task.Run(() => { agent.Ask("in the background"); });
                }
                void Update(int frame) { agent.Ask("not Unity's Update"); }
            }
            public sealed class Kitchen
            {
                NeedleAgent agent;
                void Update() { agent.Ask("not a MonoBehaviour"); }
            }
            """);
        Assert.Empty(result.Needle);
        Assert.Empty(result.Errors);
    }
}
