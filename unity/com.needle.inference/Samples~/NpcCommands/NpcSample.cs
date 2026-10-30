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
            channel = host.StartTyped<NpcSink, NpcCommand>(new HostOptions { ModelPath = path, ToolsJson = tools, Session = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(2) } });
            var manager = World.DefaultGameObjectInjectionWorld.EntityManager;
            singleton = manager.CreateSingleton(new NeedleChannelHandle { Channel = channel, Identity = channel.Identity });
            npc = manager.CreateEntity(typeof(NeedlePending), typeof(NpcCommandElement));
            manager.SetComponentEnabled<NeedlePending>(npc, false);
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
