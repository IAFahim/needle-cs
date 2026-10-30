using System.Threading;
using Unity.Profiling;

namespace Needle
{
    public static unsafe class NeedleProfiler
    {
        public static readonly ProfilerMarker Submit = new ProfilerMarker(ProfilerCategory.Scripts, "Needle.Channel.Submit");
        public static readonly ProfilerMarker Receive = new ProfilerMarker(ProfilerCategory.Scripts, "Needle.Channel.Receive");
        public static readonly ProfilerMarker SystemUpdate = new ProfilerMarker(ProfilerCategory.Scripts, "Needle.System");

        static System.IntPtr attached;
        static readonly ProfilerCounterValue<long> queueDepth = new ProfilerCounterValue<long>("Needle Queue Depth");
        static readonly ProfilerCounterValue<long> requestsCompleted = new ProfilerCounterValue<long>("Needle Requests Completed");
        static readonly ProfilerCounterValue<long> requestsFailed = new ProfilerCounterValue<long>("Needle Requests Failed");
        static readonly ProfilerCounterValue<double> tokensPerSecond = new ProfilerCounterValue<double>("Needle Tokens Per Second");
        static readonly ProfilerCounterValue<double> workerBusy = new ProfilerCounterValue<double>("Needle Worker Busy");
        static readonly ProfilerCounterValue<long> prefillLastMicros = new ProfilerCounterValue<long>("Needle Prefill Last Us");
        static readonly ProfilerCounterValue<long> decodeLastMicrosPerToken = new ProfilerCounterValue<long>("Needle Decode Last Us Per Token");
        static readonly ProfilerCounterValue<double> confidenceMean = new ProfilerCounterValue<double>("Needle Confidence Mean");

        public static void Attach(NeedleStats* stats) => Volatile.Write(ref attached, (System.IntPtr)stats);

        public static NeedleStats* Attached => (NeedleStats*)Volatile.Read(ref attached);

        public static void Sample()
        {
            var stats = Attached;
            if (stats == null) return;
            queueDepth.Value = Volatile.Read(ref stats->QueueDepth);
            requestsCompleted.Value = Volatile.Read(ref stats->Completed);
            requestsFailed.Value = Volatile.Read(ref stats->Failed) + Volatile.Read(ref stats->Cancelled);
            tokensPerSecond.Value = NeedleTelemetry.TokensPerSecond(stats);
            workerBusy.Value = NeedleTelemetry.BusyFraction(stats);
            prefillLastMicros.Value = Volatile.Read(ref stats->LastPrefillMicros);
            decodeLastMicrosPerToken.Value = Volatile.Read(ref stats->LastDecodeMicrosPerToken);
            confidenceMean.Value = NeedleTelemetry.MeanConfidence(stats);
        }

        public static void Detach() => Volatile.Write(ref attached, System.IntPtr.Zero);
    }
}
