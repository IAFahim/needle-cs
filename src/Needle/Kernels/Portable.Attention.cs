using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Needle;

// N54: portable scalar attention tile (plan-netstandard.md Track C).
//
// Fallback for runtimes with no SIMD provider (IL2CPP, Mono, or any host
// that never installs a faster AttentionTile entry). Same numerical
// contract as the kernel table: the integer QK dots are bit-exact with
// Scalar, the float tail (softmax, value accumulate) is parity-bound
// (conformance cosine >= 0.99995; in practice far tighter, since the exp
// calls are the same MathF.Exp Scalar uses, in the same order).
//
// Layout mirrors Attention.GroupAvx2, not Scalar.AttentionTile: Scalar
// scores one head at a time into a flat buffer, while the vector path
// interleaves heads as scores[s * Lanes + h] so that one K row and one V
// row each feed the whole GQA group. Portable keeps that interleaving:
//   * scores: 4 positions per iteration; each K row is loaded once and
//     dotted against every head (Dot48: 48-wide int dot, unrolled 4x).
//   * values: each V row is loaded once per position and accumulated into
//     every head (mirror of AccumulateFloatSix; the full 6-head group takes
//     the unrolled path, smaller groups the generic loop; single-head tiles
//     use a flat-scores fast path (Core1) that is bit-exact with Scalar).
// Softmax, normalizers and probability capture follow Normalizers and
// CaptureProbabilities exactly, including the chunked-partials path (Peaks
// set: peaks/sums stored, deferred inverse of 1).
//
// Wiring: Kernels.Active.AttentionTile is void(AttentionTile*). The entry is
// exposed through a Cdecl delegate rooted in Holder (install-time
// allocation only; the hot path allocates nothing), so this file needs no
// UnmanagedCallersOnly polyfill and compiles identically on net10.0 and
// netstandard2.1. Install fills only an empty slot and never clobbers a
// faster provider installed earlier.
internal static unsafe class PortableAttention
{
    const int QkDim = 48;
    const int VDim = 64;
    const int Lanes = 8;
    const float InverseSqrtQkDim = 0.14433756729740643f;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate void AttentionTileEntry(AttentionTile* tile);

    static readonly AttentionTileEntry Holder = Entry;

    /// <summary>Address for Kernels.Active.AttentionTile (Cdecl, void(AttentionTile*)).</summary>
    public static nint Address => Marshal.GetFunctionPointerForDelegate(Holder);

    /// <summary>
    /// Install the portable tile when no provider claimed AttentionTile yet.
    /// Returns true when the table now points at the portable entry.
    /// </summary>
    public static bool Install(string? provider = "portable")
    {
        var table = Kernels.Active;
        if (table.AttentionTile == 0)
        {
            table.AttentionTile = Address;
            Kernels.Install(table, Kernels.Provider ?? provider);
            return true;
        }
        return false;
    }

    internal static void Run(in AttentionTile tile)
    {
        fixed (AttentionTile* p = &tile)
            Core(p);
    }

    static void Entry(AttentionTile* tile) => Core(tile);

    // Integer QK dot, exact: the products are exact for sbyte inputs and
    // integer addition commutes, so any unroll order matches the naive
    // Scalar loop bit for bit. Internal (not private) so conformance
    // benches can spot-check it against a naive loop.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int Dot48(sbyte* a, sbyte* b)
    {
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (var i = 0; i < QkDim; i += 4)
        {
            s0 += a[i] * b[i];
            s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2];
            s3 += a[i + 3] * b[i + 3];
        }
        return (s0 + s1) + (s2 + s3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void ScoreRow(sbyte* queries, int heads, float* queryScale, sbyte* key, float keyScale, float* dst)
    {
        // Same float order as Attention.Score: (float)dot * keyScale, then
        // the per-head query scale.
        for (var h = 0; h < heads; h++)
            dst[h] = Dot48(queries + (long)h * QkDim, key) * keyScale * queryScale[h];
    }

    // Single-head tiles take a flat-scores fast path (no group to share, so
    // the interleaved layout only costs): identical op order to
    // Scalar.AttentionHead, hence bit-exact, with the unrolled Dot48.
    [SkipLocalsInit]
    static void Core1(AttentionTile* tile)
    {
        var count = tile->Count;
        var keyStride = tile->KeyStride;
        var valueStride = tile->ValueStride;
        var scaleStride = tile->ScaleStride;
        var queryScale = tile->QueryScales[0] * InverseSqrtQkDim;
        var query = tile->Queries;
        var keys = tile->Keys + (long)tile->First * keyStride;
        var keyScales = tile->KeyScales + (long)tile->First * scaleStride;
        var scores = tile->Scores;
        for (var s = 0; s < count; s++)
            scores[s] = Dot48(query, keys + (long)s * keyStride) * keyScales[(long)s * scaleStride] * queryScale;
        var peak = float.NegativeInfinity;
        for (var s = 0; s < count; s++)
            peak = MathF.Max(peak, scores[s]);
        var total = 0f;
        for (var s = 0; s < count; s++)
        {
            var e = MathF.Exp(scores[s] - peak);
            scores[s] = e;
            total += e;
        }
        if (tile->Peaks == null)
        {
            for (var s = 0; s < count; s++)
                scores[s] /= total;
        }
        else
        {
            tile->Peaks[0] = peak;
            tile->Sums[0] = total;
        }
        var probabilities = tile->Probabilities;
        if (probabilities != null)
            for (var s = 0; s < count; s++)
                probabilities[tile->First + s] = scores[s];
        var values = tile->Values + (long)tile->First * valueStride;
        var valueScales = tile->ValueScales + (long)tile->First * scaleStride;
        var output = tile->Output;
        for (var d = 0; d < VDim; d++)
            output[d] = 0f;
        for (var s = 0; s < count; s++)
        {
            var weight = scores[s] * valueScales[(long)s * scaleStride];
            var row = values + (long)s * valueStride;
            for (var d = 0; d < VDim; d++)
                output[d] += weight * row[d];
        }
    }

    [SkipLocalsInit]
    static void Core(AttentionTile* tile)
    {
        if (tile->Heads == 1)
        {
            Core1(tile);
            return;
        }
        var heads = tile->Heads;
        var count = tile->Count;
        var keyStride = tile->KeyStride;
        var valueStride = tile->ValueStride;
        var scaleStride = tile->ScaleStride;

        var queryScale = stackalloc float[Lanes];
        for (var h = 0; h < heads; h++)
            queryScale[h] = tile->QueryScales[h] * InverseSqrtQkDim;

        var queries = tile->Queries;
        var keys = tile->Keys + (long)tile->First * keyStride;
        var keyScales = tile->KeyScales + (long)tile->First * scaleStride;
        var scores = tile->Scores;

        // 1. Scores, 4 positions per iteration; one K row feeds all heads.
        var blocked = count & ~3;
        var s = 0;
        for (; s < blocked; s += 4)
            for (var p = 0; p < 4; p++)
                ScoreRow(queries, heads, queryScale, keys + (long)(s + p) * keyStride,
                    keyScales[(long)(s + p) * scaleStride], scores + (long)(s + p) * Lanes);
        for (; s < count; s++)
            ScoreRow(queries, heads, queryScale, keys + (long)s * keyStride,
                keyScales[(long)s * scaleStride], scores + (long)s * Lanes);

        // 2. Per-head softmax + normalizers (mirrors Normalizers).
        var inverse = stackalloc float[Lanes];
        var peaks = tile->Peaks;
        var sums = tile->Sums;
        for (var h = 0; h < heads; h++)
        {
            var peak = float.NegativeInfinity;
            for (s = 0; s < count; s++)
                peak = MathF.Max(peak, scores[(long)s * Lanes + h]);
            var total = 0f;
            for (s = 0; s < count; s++)
            {
                var e = MathF.Exp(scores[(long)s * Lanes + h] - peak);
                scores[(long)s * Lanes + h] = e;
                total += e;
            }
            if (peaks == null)
            {
                inverse[h] = 1f / total;
            }
            else
            {
                peaks[h] = peak;
                sums[h] = total;
                inverse[h] = 1f;
            }
        }

        // 3. Probabilities (mirrors CaptureProbabilities).
        var probabilities = tile->Probabilities;
        if (probabilities != null)
        {
            var stride = tile->ProbabilityStride;
            var first = tile->First;
            for (var h = 0; h < heads; h++)
                for (s = 0; s < count; s++)
                    probabilities[(long)h * stride + first + s] = scores[(long)s * Lanes + h] * inverse[h];
        }

        // 4. Fold the value scales into the weights (mirrors the ValuesFloat
        // prologue), so the accumulate below reads one V row per position.
        var valueScales = tile->ValueScales + (long)tile->First * scaleStride;
        for (s = 0; s < count; s++)
        {
            var w = valueScales[(long)s * scaleStride];
            var row = scores + (long)s * Lanes;
            for (var h = 0; h < heads; h++)
                row[h] *= w;
        }

        // 5. Shared V accumulate (mirror of AccumulateFloatSix): one V row
        // read feeds every head; the deferred inverse applies at the store.
        // Output is write-only, so the chunked path aliasing Output onto the
        // partials slot needs no special handling.
        var values = tile->Values + (long)tile->First * valueStride;
        var output = tile->Output;
        for (var h = 0; h < heads; h++)
            for (var d = 0; d < VDim; d++)
                output[(long)h * VDim + d] = 0f;
        if (heads == 6)
            AccumulateSix(values, valueStride, scores, count, output);
        else
            AccumulateGeneric(values, valueStride, scores, count, output, heads);
        for (var h = 0; h < heads; h++)
        {
            var inv = inverse[h];
            var dst = output + (long)h * VDim;
            for (var d = 0; d < VDim; d++)
                dst[d] *= inv;
        }
    }

    // Full-group fast path: 6 explicit weight/output chains over each V row.
    static void AccumulateSix(sbyte* values, int valueStride, float* scores, int count, float* output)
    {
        var o0 = output;
        var o1 = output + VDim;
        var o2 = output + 2 * VDim;
        var o3 = output + 3 * VDim;
        var o4 = output + 4 * VDim;
        var o5 = output + 5 * VDim;
        for (var s = 0; s < count; s++)
        {
            var row = values + (long)s * valueStride;
            var w = scores + (long)s * Lanes;
            var w0 = w[0];
            var w1 = w[1];
            var w2 = w[2];
            var w3 = w[3];
            var w4 = w[4];
            var w5 = w[5];
            for (var d = 0; d < VDim; d++)
            {
                var v = (float)row[d];
                o0[d] += w0 * v;
                o1[d] += w1 * v;
                o2[d] += w2 * v;
                o3[d] += w3 * v;
                o4[d] += w4 * v;
                o5[d] += w5 * v;
            }
        }
    }

    static void AccumulateGeneric(sbyte* values, int valueStride, float* scores, int count, float* output, int heads)
    {
        for (var s = 0; s < count; s++)
        {
            var row = values + (long)s * valueStride;
            var w = scores + (long)s * Lanes;
            for (var d = 0; d < VDim; d++)
            {
                var v = (float)row[d];
                for (var h = 0; h < heads; h++)
                    output[(long)h * VDim + d] += w[h] * v;
            }
        }
    }
}
