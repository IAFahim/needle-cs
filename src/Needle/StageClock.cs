using System.Runtime.CompilerServices;

namespace Needle;

public enum Stage
{
    Embed,
    HyperPrepare,
    HyperPhi,
    HyperMix,
    Engram,
    AttentionPrep,
    AttentionQkvg,
    AttentionPostProjection,
    AttentionCore,
    AttentionGate,
    AttentionOut,
    AttentionPostNorm,
    Cross,
    Mlp,
    Combine,
    Logits,
}

public interface IStageClock
{
    void Begin();
    void Lap(Stage stage);
    void PoolRun();
}

public readonly struct NoClock : IStageClock
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Begin()
    {
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Lap(Stage stage)
    {
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PoolRun()
    {
    }
}
