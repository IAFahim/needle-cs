using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Needle.Tools;
using Xunit;

namespace Needle.Tests.Bughunt;

public sealed class ApiGeneratorBughuntTests
{
    static readonly Lazy<IIncrementalGenerator> Generator = new(() =>
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory != null && !File.Exists(Path.Combine(directory.FullName, "src", "Needle.Generators", "Needle.Generators.csproj")))
            directory = directory.Parent;
        var bin = Path.Combine(directory!.FullName, "src", "Needle.Generators", "bin");
        var path = Directory.GetFiles(bin, "Needle.Generators.dll", SearchOption.AllDirectories).OrderByDescending(File.GetLastWriteTimeUtc).First();
        var type = Assembly.LoadFrom(path).GetType("Needle.Generators.NeedleToolsGenerator", throwOnError: true)!;
        return (IIncrementalGenerator)Activator.CreateInstance(type)!;
    });

    static (string[] Generator, string[] Compiler) Run(string source, LanguageVersion language = LanguageVersion.Preview)
    {
        var references = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator)
            .Select(p => (MetadataReference)MetadataReference.CreateFromFile(p)).ToList();
        references.Add(MetadataReference.CreateFromFile(typeof(NeedleToolsAttribute).Assembly.Location));
        var parse = new CSharpParseOptions(language);
        var compilation = CSharpCompilation.Create("probe", new[] { CSharpSyntaxTree.ParseText(source, parse) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary).WithAllowUnsafe(true));
        CSharpGeneratorDriver.Create(new[] { Generator.Value.AsSourceGenerator() }, parseOptions: parse)
            .RunGeneratorsAndUpdateCompilation(compilation, out var output, out var generatorDiagnostics);
        return (generatorDiagnostics.Select(d => d.ToString()).ToArray(),
            output.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).Select(d => d.ToString()).ToArray());
    }

    static void AssertCompiles(string source)
    {
        var (generator, compiler) = Run(source);
        Assert.Empty(generator);
        Assert.Empty(compiler);
    }

    [Fact]
    public void Generator_CSharp10OnAToolSetCapableRuntime_Compiles()
    {
        var (generator, compiler) = Run("""
            using Needle.Tools;
            namespace N
            {
                [NeedleTools] public partial interface IA { [Tool("x")] void Go(int count); }
            }
            """, LanguageVersion.CSharp10);
        Assert.Empty(generator);
        Assert.Empty(compiler);
    }

    [Fact]
    public void Generator_NestedInterface_GeneratesCompilableCode() => AssertCompiles("""
        using Needle.Tools;
        namespace N;
        public static partial class Outer { [NeedleTools] public partial interface IA { [Tool("x")] void Go(int count); } }
        public static class Use { public static int N => ACommands.ToolCount; }
        """);

    [Fact]
    public void Generator_SameInterfaceNameInTwoNamespaces_GeneratesBoth() => AssertCompiles("""
        using Needle.Tools;
        namespace Game { [NeedleTools] public partial interface ICommands { [Tool("x")] void Go(int count); } }
        namespace Ui { [NeedleTools] public partial interface ICommands { [Tool("y")] void Click(int x); } }
        namespace Use { public static class P { public static int N => Game.Commands.ToolCount + Ui.Commands.ToolCount; } }
        """);

    [Fact]
    public void Generator_EnumParameterInGlobalNamespace_Compiles() => AssertCompiles("""
        using Needle.Tools;
        public enum Mode { Fast, Slow }
        [NeedleTools] public partial interface IA { [Tool("x")] void Go(Mode mode = Mode.Slow); }
        """);

    [Fact]
    public void Generator_TwoEnumsWithTheSameSimpleName_Compile() => AssertCompiles("""
        using Needle.Tools;
        namespace A { public enum Mode { X, Y } }
        namespace B { public enum Mode { P, Q } }
        namespace N { [NeedleTools] public partial interface IA { [Tool("x")] void Go(A.Mode a, B.Mode b); } }
        """);

    [Fact]
    public void Generator_ToolMethodNamedWithAKeyword_Compiles() => AssertCompiles("""
        using Needle.Tools;
        namespace N;
        [NeedleTools] public partial interface IA { [Tool("x")] void @lock(int @class); }
        """);

    [Fact]
    public void Generator_NaNAndInfinityDefaults_Compile() => AssertCompiles("""
        using Needle.Tools;
        namespace N;
        [NeedleTools] public partial interface IA { [Tool("x")] void Go(float a = float.NaN, double b = double.PositiveInfinity, double c = double.NegativeInfinity); }
        """);

    [Fact]
    public void Generator_ThirtyTwoParameters_Compile() => AssertCompiles("""
        using Needle.Tools;
        namespace N;
        [NeedleTools] public partial interface IA { [Tool("x")] void Go(int a0,int a1,int a2,int a3,int a4,int a5,int a6,int a7,int a8,int a9,int a10,int a11,int a12,int a13,int a14,int a15,int a16,int a17,int a18,int a19,int a20,int a21,int a22,int a23,int a24,int a25,int a26,int a27,int a28,int a29,int a30,int a31); }
        """);

    [Fact]
    public void Generator_DescriptionWithUnicodeLineSeparators_Compiles() => AssertCompiles("""
        using Needle.Tools;
        namespace N;
        [NeedleTools] public partial interface IA { [Tool("one\x2028two\x2029three")] void Go([Param("a\x0085b")] int count); }
        """);

    [Fact]
    public void Generator_InterfaceNotUsingTheIPrefix_KeepsItsFirstLetter() => AssertCompiles("""
        using Needle.Tools;
        namespace N;
        [NeedleTools] public partial interface Input { [Tool("x")] void Go(int count); }
        public static class P { public static int N => InputCommands.ToolCount; }
        """);

    [Fact]
    public void Generator_ParametersCollidingAfterSnakeCase_ReportAGeneratorDiagnostic()
    {
        var (generator, compiler) = Run("""
            using Needle.Tools;
            namespace N;
            [NeedleTools] public partial interface IA { [Tool("x")] void Go(int cityName, int city_name); }
            """);
        Assert.Contains(generator, d => d.Contains("NDG"));
        Assert.DoesNotContain(compiler, d => d.Contains("IATools.g.cs"));
    }

    [Fact]
    public void Generator_ToolsNamedLikeGeneratedMembers_ReportAGeneratorDiagnostic()
    {
        var (generator, compiler) = Run("""
            using Needle.Tools;
            namespace N;
            [NeedleTools] public partial interface IA { [Tool("x")] void None(); [Tool("y")] void Kind(); [Tool("z")] void Presence(); }
            """);
        Assert.Equal(3, generator.Count(d => d.Contains("NDG")));
        Assert.DoesNotContain(compiler, d => d.Contains("IATools.g.cs"));
    }
}
