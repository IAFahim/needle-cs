using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Needle.Tools;
using UnityEngine;

namespace Needle.Samples.SquadCommander
{
    public sealed class SquadCommander : MonoBehaviour
    {
        public string ModelFile = "needle3.cact";
        public Transform Squad;
        public float Speed = 6f;
        [Tooltip("Calls below this confidence are refused. Games use 0.3; the engine default is upstream 4.x's 0.1.")]
        [Range(0f, 1f)] public float MinConfidence = 0.3f;
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
                Session = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(ResolveThreads()), RetrievalTopK = SquadCommands.GrammarToolCount, RefuseBelow = MinConfidence },
            });
        }

        int ResolveThreads() => Threads > 0 ? Math.Max(1, Math.Min(4, Threads)) : DefaultThreads();

        static int DefaultThreads()
        {
            int cores = Math.Max(1, Environment.ProcessorCount);
            return Math.Max(1, Math.Min(4, cores / 2));
        }

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
                if (!hasPending || reply.Ticket != pending) continue;
                hasPending = false;
                LastLatencyFrames = Time.frameCount - submittedFrame;
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
            public void Attack(Utf8Text64 target) {  }
            public void GiveItem(Item item, int count, Utf8Text32 to) {  }
            public void SetFormation(Formation formation) => Owner.CurrentFormation = formation;
            public void Follow(Utf8Text32 target) { Owner.moving = false; }
            public void Stop() { Owner.moving = false; }
        }
    }
}
