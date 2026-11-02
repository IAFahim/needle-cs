using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Needle.Tools;
using UnityEngine;

namespace Needle.Samples.SquadCommander
{
    /// <summary>
    /// Free-form orders for a squad, from a text box or a voice transcript, with nothing
    /// allocated or blocked on the main thread. Put it on any GameObject, point <see cref="Squad"/>
    /// at the transform to move, and call <see cref="Say"/> from your UI.
    ///
    /// The model runs on NeedleHost's own worker thread. Update() only does a ring-buffer pop and
    /// copies a small struct: no JSON, no strings, no GC, no waiting. Orders are independent
    /// (StatelessQueries), and a newer order cancels one the model has not answered yet.
    /// </summary>
    public sealed class SquadCommander : MonoBehaviour
    {
        public string ModelFile = "needle3.cact";
        public Transform Squad;
        public float Speed = 6f;
        [Range(0f, 1f)] public float MinConfidence = 0.1f;
        [Tooltip("Needle worker threads. 0 = automatic: half the logical cores, capped at 4 (4 threads where the box has headroom, 2 on a 4-core laptop). Set 1-4 to force a count.")]
        public int Threads = 0;

        NeedleHost host;
        NeedleChannel channel;
        readonly byte[] query = new byte[ChannelLimits.QueryCapacity];
        ulong pending;
        bool hasPending;
        Vector3 destination;
        bool moving;

        public Formation CurrentFormation { get; private set; }
        public int LastLatencyFrames { get; private set; }
        int submittedFrame;

        void OnEnable()
        {
            var path = Path.Combine(Application.streamingAssetsPath, ModelFile);
            if (!File.Exists(path)) { Debug.LogError($"Needle model missing: {path}"); enabled = false; return; }
            host = new NeedleHost();
            channel = host.StartTyped<SquadSink, SquadCommand>(new HostOptions
            {
                ModelPath = path,
                ToolsJson = Encoding.UTF8.GetString(SquadCommands.Schema.ToArray()),
                StatelessQueries = true,
                // Keep every tool in context: the typed path has no tool retrieval, which
                // otherwise starts above RetrievalTopK (5) tools.
                // Thread budget (see Documentation~/thread-budget.md): Game workers run
                // BelowNormal and park after 50 us, and never pin cores, so Unity's job
                // workers, render thread and main thread keep their cores. Threads = 0
                // means half the logical cores capped at 4: 4 threads where the box has
                // headroom, 2 on a 4-core laptop.
                Session = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(ResolveThreads()), RetrievalTopK = SquadCommands.GrammarToolCount },
            });
        }

        int ResolveThreads() => Threads > 0 ? Math.Max(1, Math.Min(4, Threads)) : DefaultThreads();

        // Slow runtimes get 2 threads, roomy boxes 4 (N25). Deliberately NOT
        // cores-minus-Unity-job-workers: JobWorkerCount counts parked threads, not
        // busy cores — this 32-core Editor reports 31 workers, which would collapse
        // the default to 1 thread (and in Unity 7000 JobsUtility lives in
        // UnityEngine.ManagedKernelModule, so the old "Unity.Jobs" assembly lookup
        // misses everywhere). Coexistence with Unity comes from the Game policy
        // instead: BelowNormal priority, 50 us park, no pinning. Measured frame
        // impact is identical at 2 and 4 threads; see thread-budget.md.
        static int DefaultThreads()
        {
            int cores = Math.Max(1, Environment.ProcessorCount);
            return Math.Max(1, Math.Min(4, cores / 2));
        }

        /// <summary>Submit an order. Allocation-free; returns false if the text is empty, too long or the queue is full.</summary>
        public bool Say(string order)
        {
            if (host == null || string.IsNullOrEmpty(order)) return false;
            if (Encoding.UTF8.GetByteCount(order) > query.Length) return false;
            var length = Encoding.UTF8.GetBytes(order, 0, order.Length, query, 0);
            if (hasPending) host.Cancel(pending);
            if (!channel.TrySubmitQuery(new ReadOnlySpan<byte>(query, 0, length), out pending)) { hasPending = false; return false; }
            hasPending = true;
            submittedFrame = Time.frameCount;
            return true;
        }

        void Update()
        {
            if (host == null) return;

            while (channel.TryReceive(out var reply))
            {
                if (!hasPending || reply.Ticket != pending) continue; // an order we already replaced
                hasPending = false;
                LastLatencyFrames = Time.frameCount - submittedFrame;
                // Refused replies still carry their calls; act only on answers the model is sure of.
                if (reply.Reply.Status != ReplyStatus.Called || reply.Reply.Confidence < MinConfidence) continue;
                var commands = MemoryMarshal.Cast<byte, SquadCommand>(Utf8.AsSpan(ref reply.Calls, reply.Reply.Utf8Length));
                var handler = new Handler { Owner = this };
                for (var i = 0; i < commands.Length; i++) SquadCommands.Dispatch(ref handler, commands[i]);
            }

            if (moving && Squad != null)
            {
                Squad.position = Vector3.MoveTowards(Squad.position, destination, Speed * Time.deltaTime);
                moving = Squad.position != destination;
            }
        }

        void OnDisable()
        {
            host?.Dispose();
            host = null;
        }

        struct Handler : ISquadCommands
        {
            public SquadCommander Owner;
            public void MoveTo(float x, float y) { Owner.destination = new Vector3(x, 0f, y); Owner.moving = true; }
            public void Attack(Utf8Text64 target) { /* hook your combat code here; target.Span is the UTF-8 name */ }
            public void GiveItem(Item item, int count, Utf8Text32 to) { /* hook your inventory code here */ }
            public void SetFormation(Formation formation) => Owner.CurrentFormation = formation;
            public void Follow(Utf8Text32 target) { Owner.moving = false; }
            public void Stop() { Owner.moving = false; }
        }
    }
}
