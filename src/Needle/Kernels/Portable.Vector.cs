using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Needle;

// N50: portable Vector<T> provider kernels (plan-netstandard.md Track C).
//
// Runs inside the netstandard2.1 core DLL: no System.Runtime.Intrinsics (Unity
// compiles scripts at C# 9 against the .NET Standard surface, and the net10
// SIMD assembly is unreferenceable from Unity scripts). Enabled only when
// Vector.IsHardwareAccelerated is true. Every entry checks Available first
// and runs the Scalar reference when false, so the methods are safe to call
// unconditionally; the table installer should only install when Available.
//
// Scope (deliberate): float row ops + MLP elementwise that map cleanly onto
// elementwise Vector<T> arithmetic (RMS family, softmax, dot, linear mixes,
// rope, taps, post-norm, FakeA8 grid snaps, and the Kron two-phase AXPY which
// needs no byte shuffle). NOT attempted here:
//   - MatMulRows unpack (Vector<T> has no shuffle/multiply-add-unpack) — N51/N52.
//   - Prepare planes, GatherRow dequant LUTs — byte-shuffle bound — N51/N52.
//   - AttentionTile / MergePartials / whole HadamardMlp entries — N54 / N40.
//   - Sinkhorn4x4 (fixed 16 elements: call overhead dominates) and
//     ArgMaxMasked (sparse bit-driven branches) — Vector buys nothing.
//   - Sigmoid-bound elementwise (Silu/Glu/MultiplyBySigmoid): these stay on
//     Scalar. The vector minimax Exp core (ExpV) now covers Softmax,
//     GateFakeA8 and BestProbability; Silu/Glu are outside the N50
//     parenthetical (RMS, softmax, dot, Kron) and are left for N53/N54.
//
// Contract (matches section 2 provider rules): pure functions, no heap
// allocation (only tiny stackalloc scratch in Kron/QuantizeInt8), pointer +
// scalar arguments only, safe from any thread. Signatures mirror the
// KernelTable entries in Kernels/KernelTable.cs so integration wires them 1:1.
//
// Correctness: float kernels meet the N13 parity bound (cosine >= 0.99995 vs
// Scalar, same argmax); several are bit-exact by construction (same per-lane
// op order as Scalar, no FMA contraction). QuantizeInt8 is bit-exact
// (identical rounding, clamping and cast lane by lane).
public static unsafe class PortableVector
{
    /// True when Vector&lt;T&gt; lowers to hardware SIMD on this runtime.
    /// Install the portable table entries only when this is true.
    public static bool Available => Vector.IsHardwareAccelerated;

    /// Float lane count on this runtime (8 where AVX2 is up, CoreCLR N11a).
    public static int Width => Vector<float>.Count;

    const float RmsEpsilon = 1e-6f;
    const float RoundBias = 12582912f; // 1.5 * (1 << 23); matches Scalar.RoundEven

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector<float> Load(float* p, int i) =>
        Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)(p + i));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Store(float* p, int i, Vector<float> v) =>
        Unsafe.WriteUnaligned(ref *(byte*)(p + i), v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float HMax(Vector<float> v)
    {
        var m = v[0];
        for (var k = 1; k < Vector<float>.Count; k++) m = MathF.Max(m, v[k]);
        return m;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float HSum(Vector<float> v)
    {
        var s = 0f;
        for (var k = 0; k < Vector<float>.Count; k++) s += v[k];
        return s;
    }

    static float SumOfSquaresV(float* x, int n)
    {
        var w = Vector<float>.Count;
        var step = 4 * w;
        var bulk = n / step * step;
        var s0 = Vector<float>.Zero;
        var s1 = Vector<float>.Zero;
        var s2 = Vector<float>.Zero;
        var s3 = Vector<float>.Zero;
        int i = 0;
        for (; i < bulk; i += step)
        {
            var a0 = Load(x, i);
            var a1 = Load(x, i + w);
            var a2 = Load(x, i + 2 * w);
            var a3 = Load(x, i + 3 * w);
            s0 += a0 * a0;
            s1 += a1 * a1;
            s2 += a2 * a2;
            s3 += a3 * a3;
        }
        for (; i + 2 * w <= n; i += 2 * w)
        {
            var a0 = Load(x, i);
            var a1 = Load(x, i + w);
            s0 += a0 * a0;
            s1 += a1 * a1;
        }
        if (i + w <= n)
        {
            var a0 = Load(x, i);
            s2 += a0 * a0;
            i += w;
        }
        var tail = 0f;
        for (; i < n; i++) tail += x[i] * x[i];
        return HSum(s0 + s1 + s2 + s3) + tail;
    }

    static float AbsMaxV(float* x, int n)
    {
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        var peak = Vector<float>.Zero;
        int i = 0;
        for (; i < bulk; i += w) peak = Vector.Max(peak, Vector.Abs(Load(x, i)));
        var p = HMax(peak);
        for (; i < n; i++) p = MathF.Max(p, MathF.Abs(x[i]));
        return p;
    }

    static float InverseRmsV(float* x, int n) =>
        1f / MathF.Sqrt(SumOfSquaresV(x, n) / n + RmsEpsilon);

    static void FinishFakeA8(float* x, int n, float peak)
    {
        var scale = peak > 0f ? peak / 127f : 1f;
        var inv = new Vector<float>(1f / scale);
        var sc = new Vector<float>(scale);
        var bias = new Vector<float>(RoundBias);
        var low = new Vector<float>(-128f);
        var high = new Vector<float>(127f);
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        int i = 0;
        for (; i < bulk; i += w)
        {
            var v = Load(x, i) * inv;
            var q = Vector.Min(Vector.Max((v + bias) - bias, low), high);
            Store(x, i, q * sc);
        }
        for (; i < n; i++) x[i] = MathF.Min(MathF.Max(((x[i] * (1f / scale)) + RoundBias) - RoundBias, -128f), 127f) * scale;
    }

    // Vector exp: same degree-5 minimax core as Ops.Exp (net10 AVX), with
    // mul+add in place of FMA and the 2^n ldexp done with integer Vector
    // lanes. A few ulp from MathF.Exp; parity-checked (cosine + argmax), not
    // bit-exact. Arguments here are clamped, and |n| <= 128 < 2^22, so the
    // RoundEven magic rounds exactly like RoundToNearestInteger.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector<float> ExpV(Vector<float> x)
    {
        // Clamp without Vector.Min/Max (those lower to NaN-propagating ~10
        // instruction sequences; inputs here are finite, so compare+select
        // is equivalent at 2 instructions each).
        var lo = new Vector<float>(-87.3f);
        var hi = new Vector<float>(88.7f);
        var xc = Vector.ConditionalSelect(Vector.GreaterThan(x, lo), x, lo);
        xc = Vector.ConditionalSelect(Vector.GreaterThan(hi, xc), xc, hi);
        var n = ((xc * new Vector<float>(1.44269504f)) + new Vector<float>(RoundBias)) - new Vector<float>(RoundBias);
        var r = xc - (n * new Vector<float>(0.693359375f));
        r = r - (n * new Vector<float>(-2.12194440e-4f));
        var p = new Vector<float>(1.9875691500e-4f);
        p = (p * r) + new Vector<float>(1.3981999507e-3f);
        p = (p * r) + new Vector<float>(8.3334519073e-3f);
        p = (p * r) + new Vector<float>(4.1665795894e-2f);
        p = (p * r) + new Vector<float>(1.6666665459e-1f);
        p = (p * r) + new Vector<float>(5.0000001201e-1f);
        var y = ((p * (r * r)) + r) + new Vector<float>(1f);
        var bits = (Vector.ConvertToInt32(n) + new Vector<int>(127)) * new Vector<int>(1 << 23);
        return y * Vector.AsVectorSingle(bits);
    }

    // Table wiring: void(float*, float*, int) -> Dot.
    public static float Dot(float* a, float* b, int n)
    {
        if (!Available) return Scalar.Dot(a, b, n);
        var w = Vector<float>.Count;
        var step = 4 * w;
        var bulk = n / step * step;
        var s0 = Vector<float>.Zero;
        var s1 = Vector<float>.Zero;
        var s2 = Vector<float>.Zero;
        var s3 = Vector<float>.Zero;
        int i = 0;
        for (; i < bulk; i += step)
        {
            s0 += Load(a, i) * Load(b, i);
            s1 += Load(a, i + w) * Load(b, i + w);
            s2 += Load(a, i + 2 * w) * Load(b, i + 2 * w);
            s3 += Load(a, i + 3 * w) * Load(b, i + 3 * w);
        }
        for (; i + 2 * w <= n; i += 2 * w)
        {
            s0 += Load(a, i) * Load(b, i);
            s1 += Load(a, i + w) * Load(b, i + w);
        }
        if (i + w <= n)
        {
            s2 += Load(a, i) * Load(b, i);
            i += w;
        }
        var tail = 0f;
        for (; i < n; i++) tail += a[i] * b[i];
        return HSum(s0 + s1 + s2 + s3) + tail;
    }

    // Table wiring: void(float*, int, float*) -> RmsUnit. Parity bound.
    public static void RmsUnit(float* x, int n, float* destination)
    {
        if (!Available)
        {
            var fallback = 1f / MathF.Sqrt(Scalar.SumOfSquares(x, n) / n + RmsEpsilon);
            for (var j = 0; j < n; j++) destination[j] = x[j] * fallback;
            return;
        }
        var inv = new Vector<float>(InverseRmsV(x, n));
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        int i = 0;
        for (; i < bulk; i += w) Store(destination, i, Load(x, i) * inv);
        for (; i < n; i++) destination[i] = x[i] * inv[0];
    }

    // Table wiring: void(float*, float*, int, float*) -> ZeroCenteredRms. Parity bound.
    public static void ZeroCenteredRms(float* x, float* scale, int n, float* destination)
    {
        if (!Available)
        {
            var fallback = 1f / MathF.Sqrt(Scalar.SumOfSquares(x, n) / n + RmsEpsilon);
            for (var j = 0; j < n; j++) destination[j] = (1f + scale[j]) * x[j] * fallback;
            return;
        }
        var inv = new Vector<float>(InverseRmsV(x, n));
        var one = new Vector<float>(1f);
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        int i = 0;
        for (; i < bulk; i += w)
            Store(destination, i, ((one + Load(scale, i)) * Load(x, i)) * inv);
        for (; i < n; i++) destination[i] = (1f + scale[i]) * x[i] * inv[0];
    }

    // Table wiring: void(float*, int) -> Softmax. Parity bound (same argmax:
    // the max reduction is exact; the Exp is the vector minimax core).
    public static void Softmax(float* s, int n)
    {
        if (!Available)
        {
            Scalar.Softmax(s, n);
            return;
        }
        if (n <= 0) return;
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        var peak = new Vector<float>(float.NegativeInfinity);
        int i = 0;
        for (; i < bulk; i += w) peak = Vector.Max(peak, Load(s, i));
        var p = HMax(peak);
        for (; i < n; i++) p = MathF.Max(p, s[i]);
        var shift = new Vector<float>(p);
        var acc = Vector<float>.Zero;
        i = 0;
        for (; i < bulk; i += w)
        {
            var v = ExpV(Load(s, i) - shift);
            Store(s, i, v);
            acc += v;
        }
        var total = HSum(acc);
        for (; i < n; i++)
        {
            s[i] = MathF.Exp(s[i] - p);
            total += s[i];
        }
        var inv = new Vector<float>(1f / total);
        i = 0;
        for (; i < bulk; i += w) Store(s, i, Load(s, i) * inv);
        for (; i < n; i++) s[i] /= total;
    }

    // Table wiring: float(float*, int, float) -> BestProbability. Parity bound.
    public static float BestProbability(float* values, int count, float best)
    {
        if (!Available)
        {
            var fallback = 0f;
            for (var j = 0; j < count; j++) fallback += MathF.Exp(values[j] - best);
            return 1f / fallback;
        }
        var w = Vector<float>.Count;
        var bulk = count / w * w;
        var shift = new Vector<float>(best);
        var acc = Vector<float>.Zero;
        int i = 0;
        for (; i < bulk; i += w) acc += ExpV(Load(values, i) - shift);
        var total = HSum(acc);
        for (; i < count; i++) total += MathF.Exp(values[i] - best);
        return 1f / total;
    }

    // Table wiring: void(float*, int, int, float*, float*) -> Rope. Parity bound.
    public static void Rope(float* x, int heads, int headDim, float* cos, float* sin)
    {
        if (!Available)
        {
            Scalar.Rope(x, heads, headDim, cos, sin);
            return;
        }
        var half = headDim / 2;
        var w = Vector<float>.Count;
        var bulk = half / w * w;
        for (var h = 0; h < heads; h++)
        {
            var lo = x + h * headDim;
            var hi = lo + half;
            int i = 0;
            for (; i < bulk; i += w)
            {
                var c = Load(cos, i);
                var sn = Load(sin, i);
                var a = Load(lo, i);
                var b = Load(hi, i);
                Store(lo, i, (a * c) - (b * sn));
                Store(hi, i, (b * c) + (a * sn));
            }
            for (; i < half; i++)
            {
                var a = lo[i];
                var b = hi[i];
                lo[i] = (a * cos[i]) - (b * sin[i]);
                hi[i] = (b * cos[i]) + (a * sin[i]);
            }
        }
    }

    // Table wiring: void(float*, int, float x4, float*) -> Mix4. Bit-exact:
    // same per-lane mul/add order as Scalar, no FMA contraction.
    public static void Mix4(float* s, int dim, float w0, float w1, float w2, float w3, float* destination)
    {
        if (!Available)
        {
            Scalar.Mix4(s, dim, w0, w1, w2, w3, destination);
            return;
        }
        var v0 = new Vector<float>(w0);
        var v1 = new Vector<float>(w1);
        var v2 = new Vector<float>(w2);
        var v3 = new Vector<float>(w3);
        var w = Vector<float>.Count;
        var bulk = dim / w * w;
        int c = 0;
        for (; c < bulk; c += w)
        {
            var acc = Load(s, c) * v0;
            acc += Load(s + dim, c) * v1;
            acc += Load(s + 2 * dim, c) * v2;
            acc += Load(s + 3 * dim, c) * v3;
            Store(destination, c, acc);
        }
        for (; c < dim; c++) destination[c] = w0 * s[c] + w1 * s[dim + c] + w2 * s[2 * dim + c] + w3 * s[3 * dim + c];
    }

    // Table wiring: void(float*, int, float x4, float*, float, float*) -> Combine4.
    public static void Combine4(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination)
    {
        if (!Available)
        {
            Scalar.Combine4(s, dim, w0, w1, w2, w3, y, post, destination);
            return;
        }
        var v0 = new Vector<float>(w0);
        var v1 = new Vector<float>(w1);
        var v2 = new Vector<float>(w2);
        var v3 = new Vector<float>(w3);
        var vp = new Vector<float>(post);
        var w = Vector<float>.Count;
        var bulk = dim / w * w;
        int c = 0;
        for (; c < bulk; c += w)
        {
            var acc = vp * Load(y, c);
            acc += Load(s, c) * v0;
            acc += Load(s + dim, c) * v1;
            acc += Load(s + 2 * dim, c) * v2;
            acc += Load(s + 3 * dim, c) * v3;
            Store(destination, c, acc);
        }
        for (; c < dim; c++) destination[c] = post * y[c] + w0 * s[c] + w1 * s[dim + c] + w2 * s[2 * dim + c] + w3 * s[3 * dim + c];
    }

    // Table wiring: void(float*, int, float x4, float*, float*, float, float*) -> CombineDiff4.
    public static void CombineDiff4(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination)
    {
        if (!Available)
        {
            Scalar.CombineDiff4(s, dim, w0, w1, w2, w3, x, u, post, destination);
            return;
        }
        var v0 = new Vector<float>(w0);
        var v1 = new Vector<float>(w1);
        var v2 = new Vector<float>(w2);
        var v3 = new Vector<float>(w3);
        var vp = new Vector<float>(post);
        var w = Vector<float>.Count;
        var bulk = dim / w * w;
        int c = 0;
        for (; c < bulk; c += w)
        {
            var acc = vp * (Load(x, c) - Load(u, c));
            acc += Load(s, c) * v0;
            acc += Load(s + dim, c) * v1;
            acc += Load(s + 2 * dim, c) * v2;
            acc += Load(s + 3 * dim, c) * v3;
            Store(destination, c, acc);
        }
        for (; c < dim; c++) destination[c] = post * (x[c] - u[c]) + w0 * s[c] + w1 * s[dim + c] + w2 * s[2 * dim + c] + w3 * s[3 * dim + c];
    }

    // Table wiring: void(float*, int, float*, float*) -> Taps3.
    public static void Taps3(float* taps, int width, float* current, float* destination)
    {
        if (!Available)
        {
            Scalar.Taps3(taps, width, current, destination);
            return;
        }
        var w = Vector<float>.Count;
        var bulk = width / w * w;
        int c = 0;
        for (; c < bulk; c += w)
        {
            var acc = Load(taps, c) * Load(current, c);
            acc += Load(taps + width, c) * Load(current - width, c);
            acc += Load(taps + 2 * width, c) * Load(current - 2 * width, c);
            Store(destination, c, acc);
        }
        for (; c < width; c++) destination[c] = taps[c] * current[c] + taps[width + c] * current[c - width] + taps[2 * width + c] * current[c - 2 * width];
    }

    // Table wiring: void(float*, int, float*) -> RmsFakeA8. Parity bound
    // (bit-exact unless an input sits exactly on a rounding tie shifted by the
    // ~1ulp vector-sum ordering of the inverse).
    public static void RmsFakeA8(float* x, int n, float* destination)
    {
        if (!Available)
        {
            Scalar.RmsFakeA8(x, n, destination);
            return;
        }
        var inverse = InverseRmsV(x, n);
        var inv = new Vector<float>(inverse);
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        var peak = Vector<float>.Zero;
        int i = 0;
        for (; i < bulk; i += w)
        {
            var v = Load(x, i) * inv;
            Store(destination, i, v);
            peak = Vector.Max(peak, Vector.Abs(v));
        }
        var p = HMax(peak);
        for (; i < n; i++)
        {
            destination[i] = x[i] * inverse;
            p = MathF.Max(p, MathF.Abs(destination[i]));
        }
        FinishFakeA8(destination, n, p);
    }

    // Table wiring: void(float*, float*, int, float*) -> CenteredRmsFakeA8.
    public static void CenteredRmsFakeA8(float* x, float* scale, int n, float* destination)
    {
        if (!Available)
        {
            Scalar.CenteredRmsFakeA8(x, scale, n, destination);
            return;
        }
        var inverse = InverseRmsV(x, n);
        var inv = new Vector<float>(inverse);
        var one = new Vector<float>(1f);
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        var peak = Vector<float>.Zero;
        int i = 0;
        for (; i < bulk; i += w)
        {
            var v = ((one + Load(scale, i)) * Load(x, i)) * inv;
            Store(destination, i, v);
            peak = Vector.Max(peak, Vector.Abs(v));
        }
        var p = HMax(peak);
        for (; i < n; i++)
        {
            destination[i] = (1f + scale[i]) * x[i] * inverse;
            p = MathF.Max(p, MathF.Abs(destination[i]));
        }
        FinishFakeA8(destination, n, p);
    }

    // Table wiring: void(float*, float*, int) -> GateFakeA8. Sigmoid via the
    // vector Exp core; the grid snap is vectorized.
    public static void GateFakeA8(float* x, float* gate, int n)
    {
        if (!Available)
        {
            Scalar.GateFakeA8(x, gate, n);
            return;
        }
        var one = new Vector<float>(1f);
        var zero = Vector<float>.Zero;
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        var peak = Vector<float>.Zero;
        int i = 0;
        for (; i < bulk; i += w)
        {
            var sig = one / (one + ExpV(zero - Load(gate, i)));
            var v = Load(x, i) * sig;
            Store(x, i, v);
            peak = Vector.Max(peak, Vector.Abs(v));
        }
        var p = HMax(peak);
        for (; i < n; i++)
        {
            x[i] *= 1f / (1f + MathF.Exp(-gate[i]));
            p = MathF.Max(p, MathF.Abs(x[i]));
        }
        FinishFakeA8(x, n, p);
    }

    // Table wiring: void(float*, float*, float*, float, int) -> PostNormAdd.
    public static void PostNormAdd(float* residual, float* x, float* scale, float gate, int n)
    {
        if (!Available)
        {
            Scalar.PostNormAdd(residual, x, scale, gate, n);
            return;
        }
        var inverse = InverseRmsV(x, n);
        var inv = new Vector<float>(inverse);
        var one = new Vector<float>(1f);
        var g = new Vector<float>(gate);
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        int i = 0;
        for (; i < bulk; i += w)
        {
            var v = ((one + Load(scale, i)) * Load(x, i)) * inv;
            Store(residual, i, Load(residual, i) + (g * v));
        }
        for (; i < n; i++) residual[i] += gate * ((1f + scale[i]) * x[i] * inverse);
    }

    // Table wiring: void(float*, float*, float*, int) -> MultiplyAccumulate.
    public static void MultiplyAccumulate(float* destination, float* a, float* b, int n)
    {
        if (!Available)
        {
            Scalar.MultiplyAccumulate(destination, a, b, n);
            return;
        }
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        int i = 0;
        for (; i < bulk; i += w) Store(destination, i, Load(destination, i) + (Load(a, i) * Load(b, i)));
        for (; i < n; i++) destination[i] += a[i] * b[i];
    }

    // Table wiring: void(float*, float, float*, float, int) -> ScaleAdd.
    public static void ScaleAdd(float* destination, float scale, float* x, float xScale, int n)
    {
        if (!Available)
        {
            Scalar.ScaleAdd(destination, scale, x, xScale, n);
            return;
        }
        var s = new Vector<float>(scale);
        var xs = new Vector<float>(xScale);
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        int i = 0;
        for (; i < bulk; i += w) Store(destination, i, (s * Load(destination, i)) + (xs * Load(x, i)));
        for (; i < n; i++) destination[i] = scale * destination[i] + xScale * x[i];
    }

    // Table wiring: void(float*, float*, float, int) -> AddScaled.
    public static void AddScaled(float* destination, float* x, float scale, int n)
    {
        if (!Available)
        {
            Scalar.AddScaled(destination, x, scale, n);
            return;
        }
        var s = new Vector<float>(scale);
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        int i = 0;
        for (; i < bulk; i += w) Store(destination, i, Load(destination, i) + (s * Load(x, i)));
        for (; i < n; i++) destination[i] += scale * x[i];
    }

    // Table wiring: float(float*, int, sbyte*) -> QuantizeInt8. Bit-exact:
    // the peak is order-independent, and every lane repeats Scalar's exact
    // round/clamp/cast sequence. The bulk path narrows four rounded float
    // vectors through int/short to one sbyte vector (ConvertToInt32
    // truncates, which is exact on already-integral lanes); the tail keeps
    // the scalar cast loop.
    public static float QuantizeInt8(float* x, int n, sbyte* destination)
    {
        if (!Available) return Scalar.QuantizeInt8(x, n, destination);
        var peak = AbsMaxV(x, n);
        var scale = peak > 0f ? peak / 127f : 1f;
        var inv = 1f / scale;
        var invV = new Vector<float>(inv);
        var bias = new Vector<float>(RoundBias);
        var low = new Vector<float>(-128f);
        var high = new Vector<float>(127f);
        var w = Vector<float>.Count;
        var step = 4 * w;
        var bulk4 = n / step * step;
        int i = 0;
        for (; i < bulk4; i += step)
        {
            var c0 = Vector.Min(Vector.Max(((Load(x, i) * invV) + bias) - bias, low), high);
            var c1 = Vector.Min(Vector.Max(((Load(x, i + w) * invV) + bias) - bias, low), high);
            var c2 = Vector.Min(Vector.Max(((Load(x, i + 2 * w) * invV) + bias) - bias, low), high);
            var c3 = Vector.Min(Vector.Max(((Load(x, i + 3 * w) * invV) + bias) - bias, low), high);
            var s0 = Vector.Narrow(Vector.ConvertToInt32(c0), Vector.ConvertToInt32(c1));
            var s1 = Vector.Narrow(Vector.ConvertToInt32(c2), Vector.ConvertToInt32(c3));
            Unsafe.WriteUnaligned(ref *(byte*)(destination + i), Vector.Narrow(s0, s1));
        }
        var bulk = n / w * w;
        var tmp = stackalloc float[w];
        for (; i < bulk; i += w)
        {
            var q = Vector.Min(Vector.Max(((Load(x, i) * invV) + bias) - bias, low), high);
            Store(tmp, 0, q);
            for (var k = 0; k < w; k++) destination[i + k] = (sbyte)tmp[k];
        }
        for (; i < n; i++) destination[i] = (sbyte)MathF.Min(MathF.Max(((x[i] * inv) + RoundBias) - RoundBias, -128f), 127f);
        return scale;
    }

    // Table wiring: float(float*, int) -> FakeA8. In-place grid snap; bit-exact.
    public static float FakeA8(float* x, int n)
    {
        if (!Available)
        {
            var scalarPeak = Scalar.AbsMax(x, n);
            var scalarScale = scalarPeak > 0f ? scalarPeak / 127f : 1f;
            Scalar.RoundTrip(x, n, 1f / scalarScale, scalarScale);
            return scalarScale;
        }
        var peak = AbsMaxV(x, n);
        var scale = peak > 0f ? peak / 127f : 1f;
        var inv = 1f / scale;
        var invV = new Vector<float>(inv);
        var scV = new Vector<float>(scale);
        var bias = new Vector<float>(RoundBias);
        var low = new Vector<float>(-128f);
        var high = new Vector<float>(127f);
        var w = Vector<float>.Count;
        var bulk = n / w * w;
        int i = 0;
        for (; i < bulk; i += w)
        {
            var q = Vector.Min(Vector.Max(((Load(x, i) * invV) + bias) - bias, low), high);
            Store(x, i, q * scV);
        }
        for (; i < n; i++) x[i] = MathF.Min(MathF.Max(((x[i] * inv) + RoundBias) - RoundBias, -128f), 127f) * scale;
        return scale;
    }

    // Kron two-phase AXPY (the HadamardMlp inner kernel that fits Vector:
    // every phase is a strided AXPY over columns, no shuffle needed).
    // Same per-lane op order as Scalar.Kron, hence bit-exact. Shape matches
    // Scalar.Kron so the future HadamardMlp/Kron table entry can take it.
    public static void Kron(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
    {
        if (!Available)
        {
            Scalar.Kron(z, activeRows, left, right, rows, cols, resultRows, result);
            return;
        }
        var scratch = stackalloc float[cols];
        var w = Vector<float>.Count;
        var bulk = cols / w * w;
        for (var k = 0; k < resultRows; k++)
        {
            int c = 0;
            var zero = Vector<float>.Zero;
            for (; c < bulk; c += w) Store(scratch, c, zero);
            for (; c < cols; c++) scratch[c] = 0f;
            for (var i = 0; i < activeRows; i++)
            {
                var l = new Vector<float>(left[i * rows + k]);
                var zrow = z + i * cols;
                c = 0;
                for (; c < bulk; c += w) Store(scratch, c, Load(scratch, c) + (l * Load(zrow, c)));
                for (; c < cols; c++) scratch[c] += left[i * rows + k] * zrow[c];
            }
            var output = result + k * cols;
            c = 0;
            for (; c < bulk; c += w) Store(output, c, zero);
            for (; c < cols; c++) output[c] = 0f;
            for (var j = 0; j < cols; j++)
            {
                var s = new Vector<float>(scratch[j]);
                var rrow = right + j * cols;
                c = 0;
                for (; c < bulk; c += w) Store(output, c, Load(output, c) + (s * Load(rrow, c)));
                for (; c < cols; c++) output[c] += scratch[j] * rrow[c];
            }
        }
    }
}
