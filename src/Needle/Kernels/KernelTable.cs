namespace Needle;

// N30: kernel table ABI (plan-netstandard.md section 2).
//
// A single netstandard2.1 IL binary cannot use System.Runtime.Intrinsics
// (Unity compiles scripts at C# 9 against the .NET Standard surface, and the
// net10 DLL is unreferenceable from Unity scripts). The table lets the
// fastest available provider install unmanaged entry points once at startup;
// every dispatch site in Ops.cs, Quant.cs and Blocks.cs consults it after
// the net10 direct-SIMD branch and before the Scalar reference.
//
// ABI rules (plan section 2):
//   - Blittable arguments only: pointers to the V10 structs (Cq,
//     Activations, AttentionTile, MlpBatch), float*, sbyte*, uint*, int*,
//     int, uint, float, nint. No managed objects, strings, spans or
//     delegates cross the table.
//   - Every entry does at least about 1 us of work. Per-element calls are
//     forbidden.
//   - Entries are pure: no allocation, no static mutable state, safe to call
//     from any thread (pool workers call these concurrently).
//
// Storage note: fields are nint rather than typed `delegate* unmanaged[Cdecl]`
// fields because the pointed-to structs (Cq lives in Model.cs, owned by the
// engine) stay internal; a public struct cannot expose them in field types.
// Each field's doc comment gives the exact Cdecl signature, and every dispatch
// site forwards to a NoInlining helper that casts back to that signature for
// the call. The helper keeps the unmanaged transition (P/Invoke frame) out of
// the dispatch method, so an empty table keeps today's codegen: a load, a
// test, a not-taken branch. A zero field means "not installed": fall through
// to the reference path, exactly today's behavior.
/// <summary>
/// Blittable kernel table: one address per §2.1 entry. Zero means "not
/// installed" (fall through to the reference path). Installed once via
/// <see cref="Kernels.Install"/> before workers start; read on hot paths.
/// </summary>
public struct KernelTable
{
    // void(Cq*, Activations*, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    public nint MatMulRows;
    // void(float* x, int tokens, int cols, int bits, Activations*)
    public nint Prepare;
    // void(Cq*, int row, float* destination)
    public nint GatherRow;
    // void(AttentionTile*)
    public nint AttentionTile;
    // void(float* partials, int count, int heads, float* output)
    public nint MergePartials;
    // void(MlpBatch*, float* residual, float* scratch)
    public nint HadamardMlp;
    // void(float* x, int n, float* destination)
    public nint RmsFakeA8;
    // void(float* x, float* scale, int n, float* destination)
    public nint CenteredRmsFakeA8;
    // void(float* x, float* gate, int n)
    public nint GateFakeA8;
    // void(float* residual, float* x, float* scale, float gate, int n)
    public nint PostNormAdd;
    // void(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination)
    public nint CombineDiff4;
    // void(float* s, int dim, float w0, float w1, float w2, float w3, float* destination)
    public nint Mix4;
    // void(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination)
    public nint Combine4;
    // void(float* taps, int width, float* current, float* destination)
    public nint Taps3;
    // void(float* x, int heads, int headDim, float* cos, float* sin)
    public nint Rope;
    // void(float* x, float* scale, int n, float* destination)
    public nint ZeroCenteredRms;
    // void(float* x, int n, float* destination)
    public nint RmsUnit;
    // void(float* s, int n)
    public nint Softmax;
    // float(float* a, float* b, int n)
    public nint Dot;
    // void(float* destination, float* x, float scale, int n)
    public nint AddScaled;
    // void(float* destination, float scale, float* x, float xScale, int n)
    public nint ScaleAdd;
    // void(float* destination, float* a, float* b, int n)
    public nint MultiplyAccumulate;
    // float(float* x, int n, sbyte* destination)
    public nint QuantizeInt8;
    // float(float* x, int n)
    public nint FakeA8;
    // void(float* logits, float* result)
    public nint Sinkhorn4x4;
    // int(float* x, uint* mask, int words, int length)
    public nint ArgMaxMasked;
    // float(float* values, int count, float best)
    public nint BestProbability;

    // Phase 2 (N40) reserves a LayerMember entry here; it is not implemented.

    // True when no provider installed anything: dispatch behaves exactly as
    // before N30 (net10 direct SIMD, else the Scalar reference).
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
