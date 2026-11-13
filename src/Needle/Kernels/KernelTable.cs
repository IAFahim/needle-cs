using System.Runtime.InteropServices;

namespace Needle;

[StructLayout(LayoutKind.Sequential)]
public struct KernelTable
{
    public const int FieldCount = 27;

    public nint MatMulRows;
    public nint Prepare;
    public nint GatherRow;
    public nint AttentionTile;
    public nint MergePartials;
    public nint HadamardMlp;
    public nint RmsFakeA8;
    public nint CenteredRmsFakeA8;
    public nint GateFakeA8;
    public nint PostNormAdd;
    public nint CombineDiff4;
    public nint Mix4;
    public nint Combine4;
    public nint Taps3;
    public nint Rope;
    public nint ZeroCenteredRms;
    public nint RmsUnit;
    public nint Softmax;
    public nint Dot;
    public nint AddScaled;
    public nint ScaleAdd;
    public nint MultiplyAccumulate;
    public nint QuantizeInt8;
    public nint FakeA8;
    public nint Sinkhorn4x4;
    public nint ArgMaxMasked;
    public nint BestProbability;

    public bool IsEmpty =>
        MatMulRows == 0 && Prepare == 0 && GatherRow == 0 && AttentionTile == 0 &&
        MergePartials == 0 && HadamardMlp == 0 && RmsFakeA8 == 0 &&
        CenteredRmsFakeA8 == 0 && GateFakeA8 == 0 && PostNormAdd == 0 &&
        CombineDiff4 == 0 && Mix4 == 0 && Combine4 == 0 && Taps3 == 0 &&
        Rope == 0 && ZeroCenteredRms == 0 && RmsUnit == 0 && Softmax == 0 &&
        Dot == 0 && AddScaled == 0 && ScaleAdd == 0 && MultiplyAccumulate == 0 &&
        QuantizeInt8 == 0 && FakeA8 == 0 && Sinkhorn4x4 == 0 &&
        ArgMaxMasked == 0 && BestProbability == 0;
}
