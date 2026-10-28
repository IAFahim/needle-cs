#if NEEDLE_ENTITIES
using System.Collections.Concurrent;
using Needle;
using Unity.Collections;
using Unity.Entities;

public struct NeedleRequest : IComponentData
{
    public FixedString512Bytes Query;
}

public struct NeedlePending : IComponentData
{
}

public struct NeedleResult : IComponentData
{
    public FixedString64Bytes FunctionName;
    public FixedString512Bytes ArgumentsJson;
    public float Confidence;
}

public partial class NeedleAgentSystem : SystemBase
{
    readonly ConcurrentQueue<(Entity Entity, CallResult Result)> finished = new();
    NeedleService service;

    protected override void OnCreate()
    {
        service = new NeedleService();
        service.Start(System.IO.Path.Combine(UnityEngine.Application.streamingAssetsPath, "needle3.cact"), "[{\"name\":\"set_timer\",\"description\":\"Set a timer.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"minutes\":{\"type\":\"integer\"}},\"required\":[\"minutes\"]}}]", new NeedleOptions { Threads = 2 });
    }

    protected override void OnUpdate()
    {
        var buffer = new EntityCommandBuffer(Allocator.Temp);
        foreach (var (request, entity) in SystemAPI.Query<RefRO<NeedleRequest>>().WithNone<NeedlePending>().WithEntityAccess())
        {
            var query = request.ValueRO.Query.ToString();
            buffer.AddComponent<NeedlePending>(entity);
            service.Complete(query).ContinueWith(task => finished.Enqueue((entity, task.Result)));
        }
        while (finished.TryDequeue(out var done))
        {
            buffer.RemoveComponent<NeedlePending>(done.Entity);
            buffer.RemoveComponent<NeedleRequest>(done.Entity);
            if (done.Result.FunctionCalls.Count > 0)
                buffer.AddComponent(done.Entity, new NeedleResult
                {
                    FunctionName = done.Result.FunctionCalls[0].Name,
                    ArgumentsJson = done.Result.FunctionCalls[0].ArgumentsJson,
                    Confidence = done.Result.Confidence ?? 0f,
                });
        }
        buffer.Playback(EntityManager);
        buffer.Dispose();
    }

    protected override void OnDestroy() => service.Dispose();
}
#endif
