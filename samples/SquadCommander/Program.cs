using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Needle;

namespace Needle.Samples.SquadCommander;

// A 60 Hz game loop in which the player gives a squad free-form orders ("send everyone to the
// ridge", "form a wedge"). Inference runs on the host's own worker thread; the game thread only
// submits a query, polls for the reply and applies it. The program measures exactly that
// game-thread surface and exits non-zero if it ever allocates or eats a visible slice of a frame,
// which is the property a Unity Update() or system needs.
//
//   dotnet run -c Release --project samples/SquadCommander [model.cact]
internal static class Program
{
    const int FramesPerSecond = 60;
    const double FrameSeconds = 1.0 / FramesPerSecond;
    const double RunSeconds = 12;
    // The needle calls are a lock-free ring push, a ring pop and a small memcpy; anything near a
    // millisecond means something other than that is happening on the game thread. The gate is the
    // 99th percentile, because a single frame can lose its timeslice to the OS on a busy machine
    // (the worst frame is still printed).
    const double NeedleBudgetMs = 1.0;

    // (second the player speaks, what they say). The last two arrive about a frame apart: the second
    // one supersedes the first, so the first is cancelled instead of being applied late.
    static readonly (double At, string Say)[] Script =
    {
        (0.5, "move to 120, 45"),
        (2.5, "switch to wedge formation"),
        (4.5, "attack the orc chieftain"),
        (6.5, "give 3 potions to Lyra"),
        (8.5, "follow Mira"),
        (10.0, "move to 10, 20"),
        (10.02, "actually stop everything"),
    };

    static int Main(string[] args)
    {
        var modelPath = args.Length > 0 ? args[0] : FindModel();
        if (modelPath == null || !File.Exists(modelPath))
        {
            Console.Error.WriteLine("model not found; run tools/fetch.sh or pass the path to needle3.cact");
            return 2;
        }

        var queries = new byte[Script.Length][];
        for (var i = 0; i < Script.Length; i++) queries[i] = Encoding.UTF8.GetBytes(Script[i].Say);

        var squad = new Squad { X = 0, Y = 0 };
        var handler = new SquadHandler { Squad = squad };
        var journal = new Journal(64);

        Console.WriteLine($"loading {modelPath}");
        using var host = new NeedleHost();
        var channel = host.StartTyped<SquadSink, SquadCommand>(new HostOptions
        {
            ModelPath = modelPath,
            ToolsJson = Encoding.UTF8.GetString(SquadCommands.Schema),
            // One worker leaves the rest of the cores to the game; Game parks idle workers and
            // runs them below normal priority.
            // The typed path does not do tool retrieval, which kicks in above RetrievalTopK (5) tools,
            // so keep every tool in context. Games with many more tools should group them per scene.
            // Each order is independent, so start every one from the primed tool prefix.
            StatelessQueries = true,
            Session = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(2), RetrievalTopK = SquadCommands.GrammarToolCount },
        });

        // Warm the game-thread path once so the measured frames see steady state: JIT-ed code
        // and a touched channel. This is what a game does during its loading screen.
        Warm(channel);

        var stats = new Stats();
        var pending = new Ticket[Script.Length];
        var next = 0;
        var clock = Stopwatch.StartNew();
        var frame = 0;
        long baselineAllocated = GC.GetAllocatedBytesForCurrentThread();
        long needleAllocated = 0;

        while (clock.Elapsed.TotalSeconds < RunSeconds)
        {
            var frameStart = clock.Elapsed.TotalSeconds;
            squad.Tick((float)FrameSeconds);

            // ---- the part a game integrates: everything Needle touches on the game thread ----
            var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
            var t0 = Stopwatch.GetTimestamp();

            while (next < Script.Length && frameStart >= Script[next].At)
            {
                if (channel.TrySubmitQuery(queries[next], out var ticket))
                {
                    pending[next] = new Ticket(ticket, frame, true);
                    // A newer order makes any still-unanswered older one stale.
                    for (var older = 0; older < next; older++)
                        if (pending[older].Open && !pending[older].Done) { host.Cancel(pending[older].Id); pending[older].Done = true; stats.Cancelled++; }
                }
                next++;
            }

            while (channel.TryReceive(out var reply))
            {
                var index = Find(pending, reply.Ticket);
                if (index < 0 || pending[index].Done) continue;
                pending[index].Done = true;
                stats.Replies++;
                stats.LatencyFrames.Add(frame - pending[index].SubmittedFrame);
                journal.Add(frame, index, reply.Reply);
                if (reply.Reply.Status != ReplyStatus.Called) { stats.Unhandled++; continue; }
                var bytes = Utf8.AsSpan(ref reply.Calls, reply.Reply.Utf8Length);
                var commands = MemoryMarshal.Cast<byte, SquadCommand>(bytes);
                for (var i = 0; i < commands.Length; i++)
                {
                    SquadCommands.Dispatch(ref handler, commands[i]);
                    journal.Add(frame, index, commands[i]);
                }
            }

            var elapsedMs = (Stopwatch.GetTimestamp() - t0) * 1000.0 / Stopwatch.Frequency;
            needleAllocated += GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
            stats.NeedleMs.Add(elapsedMs);
            // ---------------------------------------------------------------------------------

            journal.Flush(squad, Script, frame / (double)FramesPerSecond);

            frame++;
            var due = frameStart + FrameSeconds;
            while (clock.Elapsed.TotalSeconds < due) Thread.Sleep(0);
        }

        // Anything the model had not answered by the end is reported, not hidden. A Refused reply
        // (confidence below NeedleOptions.RefuseBelow) still carries its calls but is not applied:
        // a game should ask the player to rephrase rather than act on a guess.
        var open = 0;
        foreach (var ticket in pending) if (ticket.Open && !ticket.Done) open++;
        var allocatedTotal = GC.GetAllocatedBytesForCurrentThread() - baselineAllocated;
        return Report(frame, stats, needleAllocated, allocatedTotal, open, squad);
    }

    static int Report(int frames, Stats stats, long needleAllocated, long allocatedTotal, int open, Squad squad)
    {
        var worst = stats.NeedleMs.Max();
        var p99 = stats.NeedleMs.Percentile(0.99);
        Console.WriteLine();
        Console.WriteLine($"frames run             : {frames} ({FramesPerSecond} Hz target)");
        Console.WriteLine($"replies / cancelled    : {stats.Replies} / {stats.Cancelled} (unanswered at exit: {open}, no tool call: {stats.Unhandled})");
        Console.WriteLine($"order latency          : median {stats.LatencyFrames.Median()} frames ({stats.LatencyFrames.Median() * 1000 / FramesPerSecond} ms), max {stats.LatencyFrames.Max()} frames");
        Console.WriteLine($"game-thread needle cost: median {stats.NeedleMs.Median():0.0000} ms, p99 {p99:0.0000} ms, worst {worst:0.0000} ms (budget {NeedleBudgetMs} ms at p99)");
        Console.WriteLine($"game-thread allocations: {needleAllocated} bytes in needle calls ({allocatedTotal} in the whole loop incl. console logging)");
        Console.WriteLine($"final squad            : {squad}");

        var failures = 0;
        if (needleAllocated != 0) { Console.Error.WriteLine("FAIL: needle calls allocated on the game thread"); failures++; }
        if (p99 > NeedleBudgetMs) { Console.Error.WriteLine($"FAIL: p99 game-thread needle cost {p99:0.000} ms exceeds {NeedleBudgetMs} ms"); failures++; }
        if (stats.Replies == 0) { Console.Error.WriteLine("FAIL: no replies were received"); failures++; }
        Console.WriteLine(failures == 0 ? "hot path OK" : "hot path NOT OK");
        return failures == 0 ? 0 : 1;
    }

    static void Warm(NeedleChannel channel)
    {
        // Two real round trips: the first primes the worker's session, the second proves the
        // steady state. The replies are applied to a throwaway squad.
        var scratch = new SquadHandler { Squad = new Squad() };
        for (var i = 0; i < 2; i++)
        {
            if (!channel.TrySubmitQuery("stop"u8, out var ticket)) continue;
            var deadline = Stopwatch.StartNew();
            var done = false;
            while (!done && deadline.Elapsed.TotalSeconds < 120)
            {
                while (channel.TryReceive(out var reply))
                {
                    if (reply.Ticket != ticket) continue;
                    var commands = MemoryMarshal.Cast<byte, SquadCommand>(Utf8.AsSpan(ref reply.Calls, reply.Reply.Utf8Length));
                    for (var c = 0; c < commands.Length; c++) SquadCommands.Dispatch(ref scratch, commands[c]);
                    done = true;
                }
                if (!done) Thread.Sleep(1);
            }
        }
    }

    static int Find(Ticket[] pending, ulong id)
    {
        for (var i = 0; i < pending.Length; i++) if (pending[i].Open && pending[i].Id == id) return i;
        return -1;
    }

    static string? FindModel()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
        {
            var candidate = Path.Combine(directory.FullName, "ref", "hf", "needle3.cact");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    struct Ticket
    {
        public ulong Id;
        public int SubmittedFrame;
        public bool Open;
        public bool Done;
        public Ticket(ulong id, int frame, bool open) { Id = id; SubmittedFrame = frame; Open = open; Done = false; }
    }

    sealed class Stats
    {
        public readonly Series NeedleMs = new();
        public readonly SeriesInt LatencyFrames = new();
        public int Replies, Cancelled, Unhandled;
    }

    // Recorded in the hot path (a struct copy into a preallocated array), printed afterwards.
    sealed class Journal
    {
        readonly (int Frame, int Order, Reply Reply, SquadCommand Command, bool IsCommand)[] entries;
        int count, printed;
        public Journal(int capacity) => entries = new (int, int, Reply, SquadCommand, bool)[capacity];

        public void Add(int frame, int order, in Reply reply)
        {
            if (count < entries.Length) entries[count++] = (frame, order, reply, default, false);
        }

        public void Add(int frame, int order, in SquadCommand command)
        {
            if (count < entries.Length) entries[count++] = (frame, order, default, command, true);
        }

        public void Flush(Squad squad, (double At, string Say)[] script, double seconds)
        {
            for (; printed < count; printed++)
            {
                var (frame, order, reply, command, isCommand) = entries[printed];
                var said = $"[{seconds,5:0.00}s frame {frame,3}] \"{script[order].Say}\"";
                Console.WriteLine(isCommand ? $"{said} -> {Squad.Describe(command)}" : $"{said} reply {reply.Status} {reply.Failure} confidence {reply.Confidence:0.00}");
            }
        }
    }

    sealed class Series
    {
        readonly double[] values = new double[(int)(RunSeconds * FramesPerSecond * 2)];
        int count;
        public void Add(double value) { if (count < values.Length) values[count++] = value; }
        public double Max() { var max = 0.0; for (var i = 0; i < count; i++) max = Math.Max(max, values[i]); return max; }
        public double Median() => Percentile(0.5);
        public double Percentile(double fraction) { var copy = values.AsSpan(0, count).ToArray(); Array.Sort(copy); return copy.Length == 0 ? 0 : copy[Math.Min(copy.Length - 1, (int)(copy.Length * fraction))]; }
    }

    sealed class SeriesInt
    {
        readonly int[] values = new int[Script.Length];
        int count;
        public void Add(int value) { if (count < values.Length) values[count++] = value; }
        public int Max() { var max = 0; for (var i = 0; i < count; i++) max = Math.Max(max, values[i]); return max; }
        public int Median() { var copy = values.AsSpan(0, count).ToArray(); Array.Sort(copy); return copy.Length == 0 ? 0 : copy[copy.Length / 2]; }
    }
}
