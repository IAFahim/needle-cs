using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace Needle.Tests;

public sealed class FormatTests
{
    const string Guidance = "Refreshing the upstream model";

    readonly record struct LockedFile(string Sha256, long Bytes, string Source, string RemotePath, string LocalPath);

    readonly record struct LockFile(Dictionary<string, string> Revisions, List<LockedFile> Files);

    readonly record struct ScriptResult(int ExitCode, string Output, string Errors);

    static string LockPath => Path.Combine(Assets.Root, "tools/ref.lock");

    static LockFile ParseLock(string path)
    {
        var revisions = new Dictionary<string, string>();
        var files = new List<LockedFile>();
        foreach (var line in File.ReadAllLines(path).Where(l => l.Length > 0))
        {
            var fields = line.Split(' ');
            if (fields[0] == "revision" && fields.Length == 3) revisions[fields[1]] = fields[2];
            else if (fields[0] == "file" && fields.Length == 6) files.Add(new LockedFile(fields[1], long.Parse(fields[2]), fields[3], fields[4], fields[5]));
            else throw new FormatException($"unrecognized lock line: {line}");
        }
        return new LockFile(revisions, files);
    }

    // fetch.sh runs under Git bash on Windows, where backslash separators in
    // the script path ($0 dirname/cd) and in NEEDLE_REF_DIR/NEEDLE_REF_LOCK
    // break file lookup, so hand bash forward-slash paths on every platform
    // (no-op on Unix). CI evidence: run 41030471519 windows-x64, every
    // fetch.sh invocation exited 1 with empty output.
    static string ForBash(string path) => path.Replace(Path.DirectorySeparatorChar, '/');

    static ScriptResult RunFetch(string argument, string refDirectory, string lockPath)
    {
        var start = new ProcessStartInfo("bash") { RedirectStandardOutput = true, RedirectStandardError = true };
        start.ArgumentList.Add(ForBash(Path.Combine(Assets.Root, "tools/fetch.sh")));
        if (argument.Length > 0) start.ArgumentList.Add(argument);
        start.Environment["NEEDLE_REF_DIR"] = ForBash(refDirectory);
        start.Environment["NEEDLE_REF_LOCK"] = ForBash(lockPath);
        using var process = Process.Start(start)!;
        var errors = process.StandardError.ReadToEndAsync();
        var output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        return new ScriptResult(process.ExitCode, output, errors.Result);
    }

    static string Hex(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    static string MakeSyntheticRef(string directory, LockFile real)
    {
        var lines = real.Revisions.Select(r => $"revision {r.Key} {r.Value}").ToList();
        foreach (var file in real.Files)
        {
            var content = Encoding.UTF8.GetBytes("synthetic " + file.LocalPath);
            var destination = Path.Combine(directory, "ref", file.LocalPath);
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            File.WriteAllBytes(destination, content);
            lines.Add($"file {Hex(content)} {content.Length} {file.Source} {file.RemotePath} {file.LocalPath}");
        }
        var lockPath = Path.Combine(directory, "ref.lock");
        File.WriteAllLines(lockPath, lines);
        return lockPath;
    }

    static string NewDirectory()
    {
        var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    static byte[] RealModelBytes()
    {
        Assert.True(File.Exists(Assets.Model), $"missing {Assets.Model}; run tools/fetch.sh");
        return File.ReadAllBytes(Assets.Model);
    }

    [Fact]
    public void RealModelReportsItsFormatAndGeometry()
    {
        using var model = Model.Open(Assets.Model, false);
        var info = model.Info;
        Assert.Equal(ModelInfo.SupportedMagic, info.Magic);
        Assert.Equal(0x05E12A84u, info.Magic);
        Assert.Equal(35_335_380L, info.FileBytes);
        Assert.Equal(581, info.TensorCount);
        Assert.Equal(28, info.CodebookEntries);
        Assert.Equal(8192, info.VocabSize);
        Assert.Equal(768, info.HiddenSize);
        Assert.Equal(20, info.LayerCount);
        Assert.Equal(12, info.HeadCount);
        Assert.Equal(2, info.KeyValueHeadCount);
        Assert.Equal(48, info.QueryKeyHeadDim);
        Assert.Equal(64, info.ValueHeadDim);
        Assert.Equal(8192, info.MaxPositions);
        Assert.Equal(1024, info.SlidingWindow);
        Assert.Equal(5, info.EngramSiteCount);
        Assert.Equal(100000f, info.RopeTheta);
        Assert.True(info.HasTokenizer);
        Assert.False(info.HasAudioHead);
        Assert.True(info.HasConfidenceHead);
        Assert.False(info.HasEmbeddingHead);
        Assert.Contains("magic 0x05E12A84", info.ToString());
    }

    [Fact]
    public void RealModelAgreesWithTheUpstreamConfig()
    {
        using var model = Model.Open(Assets.Model, false);
        using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(Assets.Root, "ref/hf/config.json")));
        var root = config.RootElement;
        Assert.Equal(root.GetProperty("vocab_size").GetInt32(), model.Info.VocabSize);
        Assert.Equal(root.GetProperty("hidden_size").GetInt32(), model.Info.HiddenSize);
        Assert.Equal(root.GetProperty("num_hidden_layers").GetInt32(), model.Info.LayerCount);
        Assert.Equal(root.GetProperty("num_attention_heads").GetInt32(), model.Info.HeadCount);
        Assert.Equal(root.GetProperty("num_key_value_heads").GetInt32(), model.Info.KeyValueHeadCount);
        Assert.Equal(root.GetProperty("qk_head_dim").GetInt32(), model.Info.QueryKeyHeadDim);
        Assert.Equal(root.GetProperty("value_head_dim").GetInt32(), model.Info.ValueHeadDim);
        Assert.Equal(root.GetProperty("max_position_embeddings").GetInt32(), model.Info.MaxPositions);
        Assert.Equal(root.GetProperty("sliding_window").GetInt32(), model.Info.SlidingWindow);
        Assert.Equal(root.GetProperty("extras").GetProperty("engram").GetProperty("sites").GetArrayLength(), model.Info.EngramSiteCount);
    }

    [Fact]
    public void LoadedModelMatchesTheLockedHash()
    {
        var locked = ParseLock(LockPath).Files.Single(f => f.LocalPath == "hf/needle3.cact");
        Assert.Equal(locked.Sha256, Assets.ModelSha256);
        Assert.Equal(locked.Bytes, new FileInfo(Assets.Model).Length);
    }

    [Fact]
    public void ShortAndEmptyInputsNameTheHeaderSize()
    {
        foreach (var bytes in new[] { Array.Empty<byte>(), new byte[100] })
        {
            var error = Assert.Throws<InvalidDataException>(() => Model.FromBytes(bytes));
            Assert.Contains($"the file has {bytes.Length} bytes", error.Message);
            Assert.Contains("196-byte header", error.Message);
            Assert.Contains(Guidance, error.Message);
        }
    }

    [Fact]
    public void WrongMagicNamesFoundAndSupportedMagic()
    {
        var bytes = new byte[4096];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, 0xDEADBEEF);
        var error = Assert.Throws<InvalidDataException>(() => Model.FromBytes(bytes));
        Assert.Contains("found magic 0xDEADBEEF", error.Message);
        Assert.Contains("0x05E12A84", error.Message);
        Assert.Contains("CONTRIBUTING.md", error.Message);
        Assert.Contains(Guidance, error.Message);
    }

    [Fact]
    public void ZeroedFileReportsMagicZero()
    {
        var error = Assert.Throws<InvalidDataException>(() => Model.FromBytes(new byte[4096]));
        Assert.Contains("found magic 0x00000000", error.Message);
    }

    [Theory]
    [InlineData(300)]
    [InlineData(100_000)]
    [InlineData(1_000_000)]
    [InlineData(35_000_000)]
    public void TruncatedModelThrowsAndNamesTheLengths(int keep)
    {
        var whole = RealModelBytes();
        var error = Assert.Throws<InvalidDataException>(() => Model.FromBytes(whole.AsSpan(0, keep).ToArray()));
        Assert.Contains("truncated", error.Message);
        Assert.Contains($"the file has {keep} bytes", error.Message);
        Assert.Contains(Guidance, error.Message);
    }

    [Fact]
    public void TruncatedModelOnDiskThrowsFromOpen()
    {
        var whole = RealModelBytes();
        var directory = NewDirectory();
        try
        {
            var path = Path.Combine(directory, "short.cact");
            File.WriteAllBytes(path, whole.AsSpan(0, 2_000_000).ToArray());
            var error = Assert.Throws<InvalidDataException>(() => Model.Open(path));
            Assert.Contains("truncated", error.Message);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void UnsupportedHeadGeometryNamesFoundAndSupportedValues()
    {
        var bytes = RealModelBytes();
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(11 * 4), 32);
        var error = Assert.Throws<NotSupportedException>(() => Model.FromBytes(bytes));
        Assert.Contains("unsupported head geometry", error.Message);
        Assert.Contains("QkDim=32", error.Message);
        Assert.Contains("QkDim=48", error.Message);
        Assert.Contains(Guidance, error.Message);
    }

    [Fact]
    public void UnsupportedConvolutionGeometryNamesFoundAndSupportedValues()
    {
        var bytes = RealModelBytes();
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(19 * 4), 5);
        var error = Assert.Throws<NotSupportedException>(() => Model.FromBytes(bytes));
        Assert.Contains("ConvTaps=5", error.Message);
        Assert.Contains("ConvTaps=3", error.Message);
    }

    [Fact]
    public void OversizedEngramLayoutIsRejectedBeforeAnyWrite()
    {
        var bytes = RealModelBytes();
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(26 * 4), 9);
        var error = Assert.Throws<NotSupportedException>(() => Model.FromBytes(bytes));
        Assert.Contains("found 9 orders", error.Message);
        Assert.Contains("at most 4 orders", error.Message);
    }

    [Fact]
    public void LockFileIsCompleteAndWellFormed()
    {
        var parsed = ParseLock(LockPath);
        foreach (var source in new[] { "needle3", "whistle", "cactus" })
            Assert.Matches("^[0-9a-f]{40}$", parsed.Revisions[source]);
        Assert.Equal(9, parsed.Files.Count);
        Assert.Equal(parsed.Files.Count, parsed.Files.Select(f => f.LocalPath).Distinct().Count());
        foreach (var file in parsed.Files)
        {
            Assert.Matches("^[0-9a-f]{64}$", file.Sha256);
            Assert.True(file.Bytes > 0);
            Assert.Contains(file.Source, parsed.Revisions.Keys);
        }
        var manifest = File.ReadAllText(Path.Combine(Assets.Root, "tools/fetch.sh"));
        foreach (var file in parsed.Files)
            Assert.Contains($"\"{file.Source}|{file.RemotePath}|{file.LocalPath}\"", manifest);
    }

    [Fact]
    public void VerifyPassesOnTheDownloadedReferenceFiles()
    {
        var result = RunFetch("--verify", Path.Combine(Assets.Root, "ref"), LockPath);
        Assert.True(result.ExitCode == 0, result.Errors);
        Assert.Contains("match", result.Output);
    }

    [Fact]
    public void VerifyPassesOnMatchingFilesAndNamesEachProblem()
    {
        var directory = NewDirectory();
        try
        {
            var lockPath = MakeSyntheticRef(directory, ParseLock(LockPath));
            var refDirectory = Path.Combine(directory, "ref");
            Assert.Equal(0, RunFetch("--verify", refDirectory, lockPath).ExitCode);

            File.AppendAllText(Path.Combine(refDirectory, "hf/needle3.cact"), "changed");
            var changed = RunFetch("--verify", refDirectory, lockPath);
            Assert.Equal(1, changed.ExitCode);
            Assert.Contains("upstream changed", changed.Errors);
            Assert.Contains("hf/needle3.cact", changed.Errors);
            Assert.Contains(Guidance, changed.Errors);

            File.Delete(Path.Combine(refDirectory, "audio/test.wav"));
            var missing = RunFetch("--verify", refDirectory, lockPath);
            Assert.Equal(1, missing.ExitCode);
            Assert.Contains("missing", missing.Errors);
            Assert.Contains("audio/test.wav", missing.Errors);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void VerifyRejectsAnIncompleteOrAbsentLock()
    {
        var directory = NewDirectory();
        try
        {
            var lockPath = MakeSyntheticRef(directory, ParseLock(LockPath));
            var refDirectory = Path.Combine(directory, "ref");
            File.WriteAllLines(lockPath, File.ReadAllLines(lockPath).Where(l => !l.EndsWith("hf/config.json")));
            var incomplete = RunFetch("--verify", refDirectory, lockPath);
            Assert.Equal(1, incomplete.ExitCode);
            Assert.Contains("no entry for hf/config.json", incomplete.Errors);

            var absent = RunFetch("--verify", refDirectory, Path.Combine(directory, "none.lock"));
            Assert.Equal(1, absent.ExitCode);
            Assert.Contains("--update-lock", absent.Errors);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void UnknownOptionPrintsUsage()
    {
        var result = RunFetch("--bogus", Path.Combine(Assets.Root, "ref"), LockPath);
        Assert.Equal(2, result.ExitCode);
        Assert.Contains("--update-lock", result.Errors);
        Assert.Contains("--verify", result.Errors);
    }

    [Fact]
    public void GoldenProvenanceFailsOnlyWhenTheRecordedHashDiffers()
    {
        var loaded = new string('a', 64);
        Assert.Null(Assets.ProvenanceFailureOf("{\"budget\":40}", loaded));
        Assert.Null(Assets.ProvenanceFailureOf($"{{\"model_sha256\":\"{loaded.ToUpperInvariant()}\"}}", loaded));
        var failure = Assets.ProvenanceFailureOf($"{{\"model_sha256\":\"{new string('b', 64)}\"}}", loaded);
        Assert.NotNull(failure);
        Assert.Contains("fixtures were generated from a different model; regenerate with tools/golden.py", failure);
        Assert.Contains(new string('b', 64), failure);
        Assert.Contains(loaded, failure);
    }
}
