using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Needle;

public sealed class InferenceThread : IDisposable
{
    readonly Queue<Action> queue = new();
    readonly Thread thread;
    bool closing;

    public InferenceThread(string name = "needle-inference")
    {
        thread = new Thread(Loop) { IsBackground = true, Name = name };
        thread.Start();
    }

    public Task<T> Run<T>(Func<T> work)
    {
        var completion = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        Enqueue(() =>
        {
            try { completion.SetResult(work()); }
            catch (Exception e) { completion.SetException(e); }
        });
        return completion.Task;
    }

    public Task Run(Action work) => Run(() => { work(); return 0; });

    void Enqueue(Action job)
    {
        lock (queue)
        {
            if (closing) throw new ObjectDisposedException(nameof(InferenceThread));
            queue.Enqueue(job);
            Monitor.Pulse(queue);
        }
    }

    void Loop()
    {
        for (;;)
        {
            Action job;
            lock (queue)
            {
                while (queue.Count == 0 && !closing) Monitor.Wait(queue);
                if (queue.Count == 0) return;
                job = queue.Dequeue();
            }
            job();
        }
    }

    public void Dispose()
    {
        lock (queue)
        {
            closing = true;
            Monitor.PulseAll(queue);
        }
        thread.Join();
    }
}

public sealed class NeedleService : IDisposable
{
    readonly InferenceThread thread = new();
    Model? model;
    NeedleSession? session;

    public Task Start(string cactPath, string toolsJson, NeedleOptions? options = null) =>
        thread.Run(() =>
        {
            model = Model.Open(cactPath);
            session = new NeedleSession(model, toolsJson, options);
        });

    public Task<CallResult> Complete(string query) => thread.Run(() => session!.Complete(query));

    public Task<CallResult> CompleteWithToolResult(string resultJson) => thread.Run(() => session!.CompleteWithToolResult(resultJson));

    public Task Reset() => thread.Run(() => session!.Reset());

    public void Dispose()
    {
        thread.Run(() =>
        {
            session?.Dispose();
            model?.Dispose();
        }).Wait();
        thread.Dispose();
    }
}

public sealed class SpeechService : IDisposable
{
    readonly InferenceThread thread = new();
    Model? model;
    Whistle? whistle;

    public Task Start(string cactPath, int threads = 1) =>
        thread.Run(() =>
        {
            model = Model.Open(cactPath);
            whistle = new Whistle(model, threads);
        });

    public Task<Transcription> Transcribe(float[] pcm, string? language = null, IReadOnlyList<string>? keywords = null, bool wordTimestamps = false) =>
        thread.Run(() => whistle!.Transcribe(pcm, language, keywords, wordTimestamps));

    public void Dispose()
    {
        thread.Run(() =>
        {
            whistle?.Dispose();
            model?.Dispose();
        }).Wait();
        thread.Dispose();
    }
}
