using System.IO;
using Needle;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using UnityEngine;

namespace Needle.Samples
{
    public struct NpcCommandElement : IBufferElementData { public NpcCommand Command; }
    struct NpcHandler : INpcCommands
    {
        public NpcCommand Last;
        public void MoveTo(float x, float y) => Last = new NpcCommand { Kind = NpcCommandKind.MoveTo, MoveTo = new MoveToArgs { X = x, Y = y } };
        public void Attack(int targetId) => Last = new NpcCommand { Kind = NpcCommandKind.Attack, Attack = new AttackArgs { TargetId = targetId } };
        public void GiveItem(Item item, int count) => Last = new NpcCommand { Kind = NpcCommandKind.GiveItem, GiveItem = new GiveItemArgs { Item = item, Count = count } };
        public void SetFormation(Formation formation) => Last = new NpcCommand { Kind = NpcCommandKind.SetFormation, SetFormation = new SetFormationArgs { Formation = formation } };
    }
    public class NpcDirector : MonoBehaviour
    {
        public string ModelFile = "needle3.cact";
        [Tooltip("Needle worker threads. 0 = automatic: half the logical cores, capped at 4 (4 threads where the box has headroom, 2 on a 4-core laptop). Set 1-4 to force a count.")]
        public int Threads = 0;
        NeedleHost host;
        NeedleChannel channel;
        Entity singleton;
        Entity npc;
        double next;
        void OnEnable()
        {
            var path = Path.Combine(Application.streamingAssetsPath, ModelFile);
            if (!File.Exists(path)) { Debug.LogError($"Needle model missing: {path}"); enabled = false; return; }
            host = new NeedleHost();
            var tools = System.Text.Encoding.UTF8.GetString(NpcCommands.Schema.ToArray());
            // Thread budget (see Documentation~/thread-budget.md): Game workers run BelowNormal and park after 50 us, and never pin cores. Threads = 0 means half the logical cores capped at 4.
            channel = host.StartTyped<NpcSink, NpcCommand>(new HostOptions { ModelPath = path, ToolsJson = tools, Session = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(ResolveThreads()) } });
            var manager = World.DefaultGameObjectInjectionWorld.EntityManager;
            singleton = manager.CreateSingleton(new NeedleChannelHandle { Channel = channel, Identity = channel.Identity });
            npc = manager.CreateEntity(typeof(NeedlePending), typeof(NpcCommandElement));
            manager.SetComponentEnabled<NeedlePending>(npc, false);
        }
        int ResolveThreads() => Threads > 0 ? System.Math.Max(1, System.Math.Min(4, Threads)) : DefaultThreads();

        // Slow runtimes get 2 threads, roomy boxes 4 (N25). Deliberately NOT
        // cores-minus-Unity-job-workers: JobWorkerCount counts parked threads, not
        // busy cores — this 32-core Editor reports 31 workers, which would collapse
        // the default to 1 thread. Coexistence with Unity comes from the Game policy
        // instead: BelowNormal priority, 50 us park, no pinning. Measured frame
        // impact is identical at 2 and 4 threads; see thread-budget.md.
        static int DefaultThreads()
        {
            int cores = System.Math.Max(1, System.Environment.ProcessorCount);
            return System.Math.Max(1, System.Math.Min(4, cores / 2));
        }
        void Update()
        {
            channel.Wake();
            if (host == null || Time.timeAsDouble < next) return;
            var manager = World.DefaultGameObjectInjectionWorld.EntityManager;
            if (manager.HasComponent<NeedleAsk>(npc)) return;
            next = Time.timeAsDouble + 1.2;
            manager.AddComponentData(npc, new NeedleAsk { Query = new FixedString512Bytes("advance to the ridge and report") });
        }
        void OnDisable()
        {
            var world = World.DefaultGameObjectInjectionWorld;
            if (world != null && singleton != Entity.Null && world.EntityManager.Exists(singleton))
                world.EntityManager.SetComponentData(singleton, default(NeedleChannelHandle));
            host?.Dispose();
            host = null;
        }
    }
    [BurstCompile]
    public partial struct NpcCommandSystem : ISystem
    {
        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var ecb = SystemAPI.GetSingleton<EndSimulationEntityCommandBufferSystem.Singleton>().CreateCommandBuffer(state.WorldUnmanaged).AsParallelWriter();
            state.Dependency = new DispatchJob { Ecb = ecb }.ScheduleParallel(SystemAPI.QueryBuilder().WithAll<NeedleOutcome, NpcCommandElement>().Build(), state.Dependency);
        }
        [BurstCompile]
        partial struct DispatchJob : IJobEntity
        {
            public EntityCommandBuffer.ParallelWriter Ecb;
            void Execute(Entity entity, [EntityIndexInQuery] int sortKey, ref DynamicBuffer<NpcCommandElement> commands, ref NeedleOutcome outcome)
            {
                commands.Clear();
                var handler = new NpcHandler();
                unsafe
                {
                    var max = ChannelLimits.CallsCapacity / sizeof(NpcCommand);
                    var count = outcome.CallCount < max ? outcome.CallCount : max;
                    var list = (NpcCommand*)UnsafeUtility.AddressOf(ref outcome.Calls);
                    for (var i = 0; i < count; i++) { NpcCommands.Dispatch(ref handler, list[i]); commands.Add(new NpcCommandElement { Command = handler.Last }); }
                }
                Ecb.RemoveComponent<NeedleOutcome>(sortKey, entity);
            }
        }
    }
}
