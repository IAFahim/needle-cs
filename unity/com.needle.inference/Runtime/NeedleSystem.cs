using Needle;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;

namespace Needle
{
    [BurstCompile]
    public partial struct NeedleSystem : ISystem
    {
        NativeParallelHashMap<ulong, Entity> tickets;
        nint bound;
        public void OnCreate(ref SystemState state)
        {
            tickets = new NativeParallelHashMap<ulong, Entity>(128, Allocator.Persistent);
            state.RequireForUpdate<NeedleChannelHandle>();
        }
        public void OnDestroy(ref SystemState state)
        {
            if (tickets.IsCreated) tickets.Dispose();
        }
        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var handle = SystemAPI.GetSingleton<NeedleChannelHandle>();
            if (handle.Identity == 0) return;
            if (handle.Identity != bound) { tickets.Clear(); bound = handle.Identity; }
            var ecb = SystemAPI.GetSingleton<EndSimulationEntityCommandBufferSystem.Singleton>();
            state.Dependency = new SubmitJob { Channel = handle.Channel, Tickets = tickets.AsParallelWriter(), Ecb = ecb.CreateCommandBuffer(state.WorldUnmanaged).AsParallelWriter() }
                .ScheduleParallel(SystemAPI.QueryBuilder().WithAll<NeedleAsk>().WithDisabled<NeedlePending>().Build(), state.Dependency);
            state.Dependency = new DrainJob { Channel = handle.Channel, Map = tickets, Ecb = ecb.CreateCommandBuffer(state.WorldUnmanaged) }.Schedule(state.Dependency);
        }
        [BurstCompile]
        partial struct SubmitJob : IJobEntity
        {
            [NativeDisableUnsafePtrRestriction] public NeedleChannel Channel;
            public NativeParallelHashMap<ulong, Entity>.ParallelWriter Tickets;
            public EntityCommandBuffer.ParallelWriter Ecb;
            void Execute(Entity entity, [EntityIndexInQuery] int sortKey, ref NeedleAsk ask)
            {
                ulong ticket;
                unsafe { byte* bytes = ask.Query.GetUnsafePtr(); if (!Channel.TrySubmitSilent(bytes, ask.Query.Length, ChannelRequestKind.Query, out ticket)) return; }
                Tickets.TryAdd(ticket, entity);
                Ecb.SetComponentEnabled<NeedlePending>(sortKey, entity, true);
            }
        }
        [BurstCompile]
        struct DrainJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public NeedleChannel Channel;
            public NativeParallelHashMap<ulong, Entity> Map;
            public EntityCommandBuffer Ecb;
            public void Execute()
            {
                while (Channel.TryReceive(out var reply))
                {
                    if (!Map.TryGetValue(reply.Ticket, out var entity)) continue;
                    Map.Remove(reply.Ticket);
                    Ecb.AddComponent(entity, new NeedleOutcome { Status = reply.Reply.Status, Confidence = reply.Reply.Confidence, CallCount = reply.Reply.CallCount, Calls = reply.Calls });
                    Ecb.SetComponentEnabled<NeedlePending>(entity, false);
                    Ecb.RemoveComponent<NeedleAsk>(entity);
                }
            }
        }
    }
}
