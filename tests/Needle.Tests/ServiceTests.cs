using System;
using System.IO;
using System.Threading.Tasks;
using Xunit;

namespace Needle.Tests;

public sealed class ServiceTests
{
    [Fact]
    public async Task NeedleServiceRunsOffTheCallerThread()
    {
        Assets.Require();
        using var service = new NeedleService();
        await service.Start(Assets.Model, File.ReadAllText(Assets.Tools), new NeedleOptions { Threads = 2 });
        var tasks = new[] { service.Complete("what's it like in Lagos right now?"), service.Complete("set a timer for 12 minutes") };
        var results = await Task.WhenAll(tasks);
        Assert.Equal("get_weather", results[0].FunctionCalls[0].Name);
        Assert.Contains(results[1].FunctionCalls, c => c.Name == "set_timer");
    }

    [Fact]
    public async Task SpeechServiceTranscribes()
    {
        var path = Path.Combine(Assets.Root, "ref/hf/whistle/whistle.cact");
        using var service = new SpeechService();
        await service.Start(path, 2);
        var result = await service.Transcribe(Wav.Read(Path.Combine(Assets.Root, "ref/audio/record.wav")));
        Assert.Contains("speed of sound", result.Text);
    }

    [Fact]
    public unsafe void ModelsLoadFromMemoryWithoutMmap()
    {
        var bytes = File.ReadAllBytes(Path.Combine(Assets.Root, "ref/hf/whistle/whistle.cact"));
        using var model = Model.FromBytes(bytes);
        using var whistle = new Whistle(model, 1);
        Assert.Contains("speed of sound", whistle.Transcribe(Wav.Read(Path.Combine(Assets.Root, "ref/audio/record.wav"))).Text);
    }

    [Fact]
    public async Task DisposedThreadRejectsNewWork()
    {
        var thread = new InferenceThread();
        Assert.Equal(7, await thread.Run(() => 7));
        thread.Dispose();
        Assert.Throws<ObjectDisposedException>(() => { _ = thread.Run(() => 1); });
    }
}
