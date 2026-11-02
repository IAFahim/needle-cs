namespace Needle;

// N51: tuned scalar matmul (portable provider inside the core DLL).
//
// Blocking mirrors Gemv's RowBlock/Pair2 design, but in plain scalar C# so it
// runs on every runtime (CoreCLR, Mono, IL2CPP) with no intrinsics:
//   - 4-row x 2-token tiles. One activation load serves all 4 rows (the 128
//     int8 values of a group are loaded once per token, not once per row) and
//     one packed-weight load serves both tokens. Register (local) accumulators
//     are held across the group loop and folded once per group.
//   - The int8 dot loop is unrolled by 4 (four j-planes per iteration).
//   - Each row's dot consumes its weight lookups immediately (no hoisted
//     weight tables): this keeps the live set small so the JIT holds dots,
//     activations and indices in registers instead of spilling to the stack.
//   - Per-(row, token) float work keeps Scalar's exact order: for each group
//     in order, scaled = HalfToSingle(norm) * byteScale * scale, then
//     accumulator += scaled * dot and bias += scaled * offset, finishing with
//     y = accumulator - bias. Integer dots cannot overflow (about 2M worst
//     case), so any summation order is bit-identical; the float order matches
//     Scalar term for term, hence bit-exact outputs.
//   - HalfToSingle is evaluated once per (row, group) and shared by both
//     tokens (pure function of the norm, same value Scalar recomputes).
//   - Row tails (< 4 rows) delegate to Scalar.MatMulRows, trivially exact.
//
// Wiring API (for the N30 owner; this file adds no dispatch itself):
//   PortableMatMul.MatMulRows(in Cq, in Activations, int tokens, float* y,
//     int ldy, int rowStart, int rowEnd) — a drop-in for Scalar.MatMulRows.
//   Suggested dispatch in Quant.MatMulRows, after the table check and before
//   the Scalar fallback (or wherever the portable provider slots in):
//     if (!Simd.Enabled) { PortableMatMul.MatMulRows(w, a, tokens, y, ldy, rowStart, rowEnd); return; }
internal static unsafe class PortableMatMul
{
    public static void MatMulRows(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    {
        if (tokens <= 0 || rowEnd <= rowStart) return;
        if (w.Bits == 2) Run2(w, a, tokens, y, ldy, rowStart, rowEnd);
        else Run4(w, a, tokens, y, ldy, rowStart, rowEnd);
    }

    static void Run2(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    {
        var pairs = tokens & ~1;
        var t = 0;
        for (; t < pairs; t += 2)
            RowsPair2(w, a, t, y + (long)t * ldy, ldy, rowStart, rowEnd);
        if (t < tokens)
            RowsSingle2(w, a, t, y + (long)t * ldy, ldy, rowStart, rowEnd);
    }

    static void Run4(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    {
        var pairs = tokens & ~1;
        var t = 0;
        for (; t < pairs; t += 2)
            RowsPair4(w, a, t, y + (long)t * ldy, ldy, rowStart, rowEnd);
        if (t < tokens)
            RowsSingle4(w, a, t, y + (long)t * ldy, ldy, rowStart, rowEnd);
    }

    static void RowsPair2(in Cq w, in Activations a, int t0, float* y0, int ldy, int rowStart, int rowEnd)
    {
        var n = rowStart;
        var quadEnd = rowStart + ((rowEnd - rowStart) & ~3);
        for (; n < quadEnd; n += 4)
            QuadPair2(w, a, t0, y0, ldy, n);
        TailRows(w, a, t0, 2, y0, ldy, n, rowEnd);
    }

    static void RowsSingle2(in Cq w, in Activations a, int t, float* y0, int ldy, int rowStart, int rowEnd)
    {
        var n = rowStart;
        var quadEnd = rowStart + ((rowEnd - rowStart) & ~3);
        for (; n < quadEnd; n += 4)
            QuadSingle2(w, a, t, y0, ldy, n);
        TailRows(w, a, t, 1, y0, ldy, n, rowEnd);
    }

    // Bit-identical copy of Scalar.HalfToSingle, forced inline: the group loop
    // calls this 4x per group per tile and the JIT leaves cross-class calls
    // uninlined in these large methods. Must stay bit-for-bit equal to the
    // Scalar reference (exact conversion, no rounding involved).
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    static float HalfToSingle(ushort half)
    {
        var magnitude = half & 0x7FFF;
        if (magnitude == 0) return 0f;
        if (magnitude < 0x0400) return (half & 0x8000) != 0 ? -magnitude * 5.9604645e-8f : magnitude * 5.9604645e-8f;
        var bits = ((uint)magnitude << 13) + 0x38000000u | (uint)(half & 0x8000) << 16;
        return *(float*)&bits;
    }

    static void RowsPair4(in Cq w, in Activations a, int t0, float* y0, int ldy, int rowStart, int rowEnd)
    {
        var n = rowStart;
        var quadEnd = rowStart + ((rowEnd - rowStart) & ~3);
        for (; n < quadEnd; n += 4)
            QuadPair4(w, a, t0, y0, ldy, n);
        TailRows(w, a, t0, 2, y0, ldy, n, rowEnd);
    }

    static void RowsSingle4(in Cq w, in Activations a, int t, float* y0, int ldy, int rowStart, int rowEnd)
    {
        var n = rowStart;
        var quadEnd = rowStart + ((rowEnd - rowStart) & ~3);
        for (; n < quadEnd; n += 4)
            QuadSingle4(w, a, t, y0, ldy, n);
        TailRows(w, a, t, 1, y0, ldy, n, rowEnd);
    }

    // Row remainder (< 4 rows): run the Scalar reference per token slice.
    static void TailRows(in Cq w, in Activations a, int t0, int count, float* y0, int ldy, int n, int rowEnd)
    {
        if (n >= rowEnd) return;
        for (var k = 0; k < count; k++)
            Scalar.MatMulRows(w, a.Slice(t0 + k), 1, y0 + (long)k * ldy, ldy, n, rowEnd);
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void QuadPair2(in Cq w, in Activations a, int t0, float* y0, int ldy, int n0)
    {
        var groups = w.Groups;
        var table = w.Book->Bytes;
        var byteScale = w.Book->ByteScale;
        var rowBytes = w.RowBytes;
        var cols = a.Cols;
        var stride = a.Stride;
        var v0 = a.Values + (long)t0 * cols;
        var v1 = v0 + cols;
        var s0 = a.Scales + (long)t0 * stride;
        var s1 = s0 + stride;
        var o0 = a.Offsets + (long)t0 * stride;
        var o1 = o0 + stride;
        var p0 = w.Indices + (long)n0 * rowBytes;
        var p1 = p0 + rowBytes;
        var p2 = p0 + 2 * rowBytes;
        var p3 = p0 + 3 * rowBytes;
        var nr0 = w.Norms + (long)n0 * groups;
        var nr1 = nr0 + groups;
        var nr2 = nr1 + groups;
        var nr3 = nr2 + groups;

        float acc00 = 0, bias00 = 0, acc01 = 0, bias01 = 0;
        float acc10 = 0, bias10 = 0, acc11 = 0, bias11 = 0;
        float acc20 = 0, bias20 = 0, acc21 = 0, bias21 = 0;
        float acc30 = 0, bias30 = 0, acc31 = 0, bias31 = 0;

        for (var g = 0; g < groups; g++)
        {
            var sc0 = s0[g];
            var oc0 = o0[g];
            var sc1 = s1[g];
            var oc1 = o1[g];
            var f0 = HalfToSingle(nr0[g]) * byteScale;
            var f1 = HalfToSingle(nr1[g]) * byteScale;
            var f2 = HalfToSingle(nr2[g]) * byteScale;
            var f3 = HalfToSingle(nr3[g]) * byteScale;
            var k00 = f0 * sc0;
            var k01 = f0 * sc1;
            var k10 = f1 * sc0;
            var k11 = f1 * sc1;
            var k20 = f2 * sc0;
            var k21 = f2 * sc1;
            var k30 = f3 * sc0;
            var k31 = f3 * sc1;

            var x0 = v0 + (long)g * Cq.GroupSize;
            var x1 = v1 + (long)g * Cq.GroupSize;
            var q0 = p0 + (long)g * 32;
            var q1 = p1 + (long)g * 32;
            var q2 = p2 + (long)g * 32;
            var q3 = p3 + (long)g * 32;

            int d00 = 0, d01 = 0, d10 = 0, d11 = 0;
            int d20 = 0, d21 = 0, d30 = 0, d31 = 0;
            for (var j = 0; j < 32; j += 4)
            {
                {
                    var jj = j;
                    int xa0 = x0[jj], xb0 = x0[jj + 32], xc0 = x0[jj + 64], xd0 = x0[jj + 96];
                    int u0 = q0[jj];
                    d00 += table[u0 & 3] * xa0 + table[u0 >> 2 & 3] * xb0 + table[u0 >> 4 & 3] * xc0 + table[u0 >> 6 & 3] * xd0;
                    int u1 = q1[jj];
                    d10 += table[u1 & 3] * xa0 + table[u1 >> 2 & 3] * xb0 + table[u1 >> 4 & 3] * xc0 + table[u1 >> 6 & 3] * xd0;
                    int u2 = q2[jj];
                    d20 += table[u2 & 3] * xa0 + table[u2 >> 2 & 3] * xb0 + table[u2 >> 4 & 3] * xc0 + table[u2 >> 6 & 3] * xd0;
                    int u3 = q3[jj];
                    d30 += table[u3 & 3] * xa0 + table[u3 >> 2 & 3] * xb0 + table[u3 >> 4 & 3] * xc0 + table[u3 >> 6 & 3] * xd0;
                    int xa1 = x1[jj], xb1 = x1[jj + 32], xc1 = x1[jj + 64], xd1 = x1[jj + 96];
                    u0 = q0[jj];
                    d01 += table[u0 & 3] * xa1 + table[u0 >> 2 & 3] * xb1 + table[u0 >> 4 & 3] * xc1 + table[u0 >> 6 & 3] * xd1;
                    u1 = q1[jj];
                    d11 += table[u1 & 3] * xa1 + table[u1 >> 2 & 3] * xb1 + table[u1 >> 4 & 3] * xc1 + table[u1 >> 6 & 3] * xd1;
                    u2 = q2[jj];
                    d21 += table[u2 & 3] * xa1 + table[u2 >> 2 & 3] * xb1 + table[u2 >> 4 & 3] * xc1 + table[u2 >> 6 & 3] * xd1;
                    u3 = q3[jj];
                    d31 += table[u3 & 3] * xa1 + table[u3 >> 2 & 3] * xb1 + table[u3 >> 4 & 3] * xc1 + table[u3 >> 6 & 3] * xd1;
                }
                {
                    var jj = j + 1;
                    int xa0 = x0[jj], xb0 = x0[jj + 32], xc0 = x0[jj + 64], xd0 = x0[jj + 96];
                    int u0 = q0[jj];
                    d00 += table[u0 & 3] * xa0 + table[u0 >> 2 & 3] * xb0 + table[u0 >> 4 & 3] * xc0 + table[u0 >> 6 & 3] * xd0;
                    int u1 = q1[jj];
                    d10 += table[u1 & 3] * xa0 + table[u1 >> 2 & 3] * xb0 + table[u1 >> 4 & 3] * xc0 + table[u1 >> 6 & 3] * xd0;
                    int u2 = q2[jj];
                    d20 += table[u2 & 3] * xa0 + table[u2 >> 2 & 3] * xb0 + table[u2 >> 4 & 3] * xc0 + table[u2 >> 6 & 3] * xd0;
                    int u3 = q3[jj];
                    d30 += table[u3 & 3] * xa0 + table[u3 >> 2 & 3] * xb0 + table[u3 >> 4 & 3] * xc0 + table[u3 >> 6 & 3] * xd0;
                    int xa1 = x1[jj], xb1 = x1[jj + 32], xc1 = x1[jj + 64], xd1 = x1[jj + 96];
                    u0 = q0[jj];
                    d01 += table[u0 & 3] * xa1 + table[u0 >> 2 & 3] * xb1 + table[u0 >> 4 & 3] * xc1 + table[u0 >> 6 & 3] * xd1;
                    u1 = q1[jj];
                    d11 += table[u1 & 3] * xa1 + table[u1 >> 2 & 3] * xb1 + table[u1 >> 4 & 3] * xc1 + table[u1 >> 6 & 3] * xd1;
                    u2 = q2[jj];
                    d21 += table[u2 & 3] * xa1 + table[u2 >> 2 & 3] * xb1 + table[u2 >> 4 & 3] * xc1 + table[u2 >> 6 & 3] * xd1;
                    u3 = q3[jj];
                    d31 += table[u3 & 3] * xa1 + table[u3 >> 2 & 3] * xb1 + table[u3 >> 4 & 3] * xc1 + table[u3 >> 6 & 3] * xd1;
                }
                {
                    var jj = j + 2;
                    int xa0 = x0[jj], xb0 = x0[jj + 32], xc0 = x0[jj + 64], xd0 = x0[jj + 96];
                    int u0 = q0[jj];
                    d00 += table[u0 & 3] * xa0 + table[u0 >> 2 & 3] * xb0 + table[u0 >> 4 & 3] * xc0 + table[u0 >> 6 & 3] * xd0;
                    int u1 = q1[jj];
                    d10 += table[u1 & 3] * xa0 + table[u1 >> 2 & 3] * xb0 + table[u1 >> 4 & 3] * xc0 + table[u1 >> 6 & 3] * xd0;
                    int u2 = q2[jj];
                    d20 += table[u2 & 3] * xa0 + table[u2 >> 2 & 3] * xb0 + table[u2 >> 4 & 3] * xc0 + table[u2 >> 6 & 3] * xd0;
                    int u3 = q3[jj];
                    d30 += table[u3 & 3] * xa0 + table[u3 >> 2 & 3] * xb0 + table[u3 >> 4 & 3] * xc0 + table[u3 >> 6 & 3] * xd0;
                    int xa1 = x1[jj], xb1 = x1[jj + 32], xc1 = x1[jj + 64], xd1 = x1[jj + 96];
                    u0 = q0[jj];
                    d01 += table[u0 & 3] * xa1 + table[u0 >> 2 & 3] * xb1 + table[u0 >> 4 & 3] * xc1 + table[u0 >> 6 & 3] * xd1;
                    u1 = q1[jj];
                    d11 += table[u1 & 3] * xa1 + table[u1 >> 2 & 3] * xb1 + table[u1 >> 4 & 3] * xc1 + table[u1 >> 6 & 3] * xd1;
                    u2 = q2[jj];
                    d21 += table[u2 & 3] * xa1 + table[u2 >> 2 & 3] * xb1 + table[u2 >> 4 & 3] * xc1 + table[u2 >> 6 & 3] * xd1;
                    u3 = q3[jj];
                    d31 += table[u3 & 3] * xa1 + table[u3 >> 2 & 3] * xb1 + table[u3 >> 4 & 3] * xc1 + table[u3 >> 6 & 3] * xd1;
                }
                {
                    var jj = j + 3;
                    int xa0 = x0[jj], xb0 = x0[jj + 32], xc0 = x0[jj + 64], xd0 = x0[jj + 96];
                    int u0 = q0[jj];
                    d00 += table[u0 & 3] * xa0 + table[u0 >> 2 & 3] * xb0 + table[u0 >> 4 & 3] * xc0 + table[u0 >> 6 & 3] * xd0;
                    int u1 = q1[jj];
                    d10 += table[u1 & 3] * xa0 + table[u1 >> 2 & 3] * xb0 + table[u1 >> 4 & 3] * xc0 + table[u1 >> 6 & 3] * xd0;
                    int u2 = q2[jj];
                    d20 += table[u2 & 3] * xa0 + table[u2 >> 2 & 3] * xb0 + table[u2 >> 4 & 3] * xc0 + table[u2 >> 6 & 3] * xd0;
                    int u3 = q3[jj];
                    d30 += table[u3 & 3] * xa0 + table[u3 >> 2 & 3] * xb0 + table[u3 >> 4 & 3] * xc0 + table[u3 >> 6 & 3] * xd0;
                    int xa1 = x1[jj], xb1 = x1[jj + 32], xc1 = x1[jj + 64], xd1 = x1[jj + 96];
                    u0 = q0[jj];
                    d01 += table[u0 & 3] * xa1 + table[u0 >> 2 & 3] * xb1 + table[u0 >> 4 & 3] * xc1 + table[u0 >> 6 & 3] * xd1;
                    u1 = q1[jj];
                    d11 += table[u1 & 3] * xa1 + table[u1 >> 2 & 3] * xb1 + table[u1 >> 4 & 3] * xc1 + table[u1 >> 6 & 3] * xd1;
                    u2 = q2[jj];
                    d21 += table[u2 & 3] * xa1 + table[u2 >> 2 & 3] * xb1 + table[u2 >> 4 & 3] * xc1 + table[u2 >> 6 & 3] * xd1;
                    u3 = q3[jj];
                    d31 += table[u3 & 3] * xa1 + table[u3 >> 2 & 3] * xb1 + table[u3 >> 4 & 3] * xc1 + table[u3 >> 6 & 3] * xd1;
                }
            }

            acc00 += k00 * d00; bias00 += k00 * oc0;
            acc01 += k01 * d01; bias01 += k01 * oc1;
            acc10 += k10 * d10; bias10 += k10 * oc0;
            acc11 += k11 * d11; bias11 += k11 * oc1;
            acc20 += k20 * d20; bias20 += k20 * oc0;
            acc21 += k21 * d21; bias21 += k21 * oc1;
            acc30 += k30 * d30; bias30 += k30 * oc0;
            acc31 += k31 * d31; bias31 += k31 * oc1;
        }

        y0[n0] = acc00 - bias00;
        y0[n0 + 1] = acc10 - bias10;
        y0[n0 + 2] = acc20 - bias20;
        y0[n0 + 3] = acc30 - bias30;
        y0[ldy + n0] = acc01 - bias01;
        y0[ldy + n0 + 1] = acc11 - bias11;
        y0[ldy + n0 + 2] = acc21 - bias21;
        y0[ldy + n0 + 3] = acc31 - bias31;
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void QuadSingle2(in Cq w, in Activations a, int t, float* y0, int ldy, int n0)
    {
        var groups = w.Groups;
        var table = w.Book->Bytes;
        var byteScale = w.Book->ByteScale;
        var rowBytes = w.RowBytes;
        var cols = a.Cols;
        var stride = a.Stride;
        var v = a.Values + (long)t * cols;
        var sc = a.Scales + (long)t * stride;
        var off = a.Offsets + (long)t * stride;
        var p0 = w.Indices + (long)n0 * rowBytes;
        var p1 = p0 + rowBytes;
        var p2 = p0 + 2 * rowBytes;
        var p3 = p0 + 3 * rowBytes;
        var nr0 = w.Norms + (long)n0 * groups;
        var nr1 = nr0 + groups;
        var nr2 = nr1 + groups;
        var nr3 = nr2 + groups;

        float acc0 = 0, bias0 = 0;
        float acc1 = 0, bias1 = 0;
        float acc2 = 0, bias2 = 0;
        float acc3 = 0, bias3 = 0;

        for (var g = 0; g < groups; g++)
        {
            var s = sc[g];
            var o = off[g];
            var f0 = HalfToSingle(nr0[g]) * byteScale;
            var f1 = HalfToSingle(nr1[g]) * byteScale;
            var f2 = HalfToSingle(nr2[g]) * byteScale;
            var f3 = HalfToSingle(nr3[g]) * byteScale;
            var k0 = f0 * s;
            var k1 = f1 * s;
            var k2 = f2 * s;
            var k3 = f3 * s;

            var x = v + (long)g * Cq.GroupSize;
            var q0 = p0 + (long)g * 32;
            var q1 = p1 + (long)g * 32;
            var q2 = p2 + (long)g * 32;
            var q3 = p3 + (long)g * 32;

            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            for (var j = 0; j < 32; j += 4)
            {
                {
                    var jj = j;
                    int xa = x[jj], xb = x[jj + 32], xc = x[jj + 64], xd = x[jj + 96];
                    int u0 = q0[jj];
                    d0 += table[u0 & 3] * xa + table[u0 >> 2 & 3] * xb + table[u0 >> 4 & 3] * xc + table[u0 >> 6 & 3] * xd;
                    int u1 = q1[jj];
                    d1 += table[u1 & 3] * xa + table[u1 >> 2 & 3] * xb + table[u1 >> 4 & 3] * xc + table[u1 >> 6 & 3] * xd;
                    int u2 = q2[jj];
                    d2 += table[u2 & 3] * xa + table[u2 >> 2 & 3] * xb + table[u2 >> 4 & 3] * xc + table[u2 >> 6 & 3] * xd;
                    int u3 = q3[jj];
                    d3 += table[u3 & 3] * xa + table[u3 >> 2 & 3] * xb + table[u3 >> 4 & 3] * xc + table[u3 >> 6 & 3] * xd;
                }
                {
                    var jj = j + 1;
                    int xa = x[jj], xb = x[jj + 32], xc = x[jj + 64], xd = x[jj + 96];
                    int u0 = q0[jj];
                    d0 += table[u0 & 3] * xa + table[u0 >> 2 & 3] * xb + table[u0 >> 4 & 3] * xc + table[u0 >> 6 & 3] * xd;
                    int u1 = q1[jj];
                    d1 += table[u1 & 3] * xa + table[u1 >> 2 & 3] * xb + table[u1 >> 4 & 3] * xc + table[u1 >> 6 & 3] * xd;
                    int u2 = q2[jj];
                    d2 += table[u2 & 3] * xa + table[u2 >> 2 & 3] * xb + table[u2 >> 4 & 3] * xc + table[u2 >> 6 & 3] * xd;
                    int u3 = q3[jj];
                    d3 += table[u3 & 3] * xa + table[u3 >> 2 & 3] * xb + table[u3 >> 4 & 3] * xc + table[u3 >> 6 & 3] * xd;
                }
                {
                    var jj = j + 2;
                    int xa = x[jj], xb = x[jj + 32], xc = x[jj + 64], xd = x[jj + 96];
                    int u0 = q0[jj];
                    d0 += table[u0 & 3] * xa + table[u0 >> 2 & 3] * xb + table[u0 >> 4 & 3] * xc + table[u0 >> 6 & 3] * xd;
                    int u1 = q1[jj];
                    d1 += table[u1 & 3] * xa + table[u1 >> 2 & 3] * xb + table[u1 >> 4 & 3] * xc + table[u1 >> 6 & 3] * xd;
                    int u2 = q2[jj];
                    d2 += table[u2 & 3] * xa + table[u2 >> 2 & 3] * xb + table[u2 >> 4 & 3] * xc + table[u2 >> 6 & 3] * xd;
                    int u3 = q3[jj];
                    d3 += table[u3 & 3] * xa + table[u3 >> 2 & 3] * xb + table[u3 >> 4 & 3] * xc + table[u3 >> 6 & 3] * xd;
                }
                {
                    var jj = j + 3;
                    int xa = x[jj], xb = x[jj + 32], xc = x[jj + 64], xd = x[jj + 96];
                    int u0 = q0[jj];
                    d0 += table[u0 & 3] * xa + table[u0 >> 2 & 3] * xb + table[u0 >> 4 & 3] * xc + table[u0 >> 6 & 3] * xd;
                    int u1 = q1[jj];
                    d1 += table[u1 & 3] * xa + table[u1 >> 2 & 3] * xb + table[u1 >> 4 & 3] * xc + table[u1 >> 6 & 3] * xd;
                    int u2 = q2[jj];
                    d2 += table[u2 & 3] * xa + table[u2 >> 2 & 3] * xb + table[u2 >> 4 & 3] * xc + table[u2 >> 6 & 3] * xd;
                    int u3 = q3[jj];
                    d3 += table[u3 & 3] * xa + table[u3 >> 2 & 3] * xb + table[u3 >> 4 & 3] * xc + table[u3 >> 6 & 3] * xd;
                }
            }

            acc0 += k0 * d0; bias0 += k0 * o;
            acc1 += k1 * d1; bias1 += k1 * o;
            acc2 += k2 * d2; bias2 += k2 * o;
            acc3 += k3 * d3; bias3 += k3 * o;
        }

        y0[n0] = acc0 - bias0;
        y0[n0 + 1] = acc1 - bias1;
        y0[n0 + 2] = acc2 - bias2;
        y0[n0 + 3] = acc3 - bias3;
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void QuadPair4(in Cq w, in Activations a, int t0, float* y0, int ldy, int n0)
    {
        var groups = w.Groups;
        var table = w.Book->Bytes;
        var byteScale = w.Book->ByteScale;
        var rowBytes = w.RowBytes;
        var cols = a.Cols;
        var stride = a.Stride;
        var v0 = a.Values + (long)t0 * cols;
        var v1 = v0 + cols;
        var s0 = a.Scales + (long)t0 * stride;
        var s1 = s0 + stride;
        var o0 = a.Offsets + (long)t0 * stride;
        var o1 = o0 + stride;
        var p0 = w.Indices + (long)n0 * rowBytes;
        var p1 = p0 + rowBytes;
        var p2 = p0 + 2 * rowBytes;
        var p3 = p0 + 3 * rowBytes;
        var nr0 = w.Norms + (long)n0 * groups;
        var nr1 = nr0 + groups;
        var nr2 = nr1 + groups;
        var nr3 = nr2 + groups;

        float acc00 = 0, bias00 = 0, acc01 = 0, bias01 = 0;
        float acc10 = 0, bias10 = 0, acc11 = 0, bias11 = 0;
        float acc20 = 0, bias20 = 0, acc21 = 0, bias21 = 0;
        float acc30 = 0, bias30 = 0, acc31 = 0, bias31 = 0;

        for (var g = 0; g < groups; g++)
        {
            var sc0 = s0[g];
            var oc0 = o0[g];
            var sc1 = s1[g];
            var oc1 = o1[g];
            var f0 = HalfToSingle(nr0[g]) * byteScale;
            var f1 = HalfToSingle(nr1[g]) * byteScale;
            var f2 = HalfToSingle(nr2[g]) * byteScale;
            var f3 = HalfToSingle(nr3[g]) * byteScale;
            var k00 = f0 * sc0;
            var k01 = f0 * sc1;
            var k10 = f1 * sc0;
            var k11 = f1 * sc1;
            var k20 = f2 * sc0;
            var k21 = f2 * sc1;
            var k30 = f3 * sc0;
            var k31 = f3 * sc1;

            var x0 = v0 + (long)g * Cq.GroupSize;
            var x1 = v1 + (long)g * Cq.GroupSize;
            var q0 = p0 + (long)g * 64;
            var q1 = p1 + (long)g * 64;
            var q2 = p2 + (long)g * 64;
            var q3 = p3 + (long)g * 64;

            int d00 = 0, d01 = 0, d10 = 0, d11 = 0;
            int d20 = 0, d21 = 0, d30 = 0, d31 = 0;
            for (var j = 0; j < 32; j += 4)
            {
                {
                    var jj = j;
                    int xa0 = x0[jj], xb0 = x0[jj + 32], xc0 = x0[jj + 64], xd0 = x0[jj + 96];
                    int b0 = q0[jj], c0 = q0[jj + 32];
                    d00 += table[b0 & 15] * xa0 + table[b0 >> 4] * xb0 + table[c0 & 15] * xc0 + table[c0 >> 4] * xd0;
                    int b1 = q1[jj], c1 = q1[jj + 32];
                    d10 += table[b1 & 15] * xa0 + table[b1 >> 4] * xb0 + table[c1 & 15] * xc0 + table[c1 >> 4] * xd0;
                    int b2 = q2[jj], c2 = q2[jj + 32];
                    d20 += table[b2 & 15] * xa0 + table[b2 >> 4] * xb0 + table[c2 & 15] * xc0 + table[c2 >> 4] * xd0;
                    int b3 = q3[jj], c3 = q3[jj + 32];
                    d30 += table[b3 & 15] * xa0 + table[b3 >> 4] * xb0 + table[c3 & 15] * xc0 + table[c3 >> 4] * xd0;
                    int xa1 = x1[jj], xb1 = x1[jj + 32], xc1 = x1[jj + 64], xd1 = x1[jj + 96];
                    b0 = q0[jj]; c0 = q0[jj + 32];
                    d01 += table[b0 & 15] * xa1 + table[b0 >> 4] * xb1 + table[c0 & 15] * xc1 + table[c0 >> 4] * xd1;
                    b1 = q1[jj]; c1 = q1[jj + 32];
                    d11 += table[b1 & 15] * xa1 + table[b1 >> 4] * xb1 + table[c1 & 15] * xc1 + table[c1 >> 4] * xd1;
                    b2 = q2[jj]; c2 = q2[jj + 32];
                    d21 += table[b2 & 15] * xa1 + table[b2 >> 4] * xb1 + table[c2 & 15] * xc1 + table[c2 >> 4] * xd1;
                    b3 = q3[jj]; c3 = q3[jj + 32];
                    d31 += table[b3 & 15] * xa1 + table[b3 >> 4] * xb1 + table[c3 & 15] * xc1 + table[c3 >> 4] * xd1;
                }
                {
                    var jj = j + 1;
                    int xa0 = x0[jj], xb0 = x0[jj + 32], xc0 = x0[jj + 64], xd0 = x0[jj + 96];
                    int b0 = q0[jj], c0 = q0[jj + 32];
                    d00 += table[b0 & 15] * xa0 + table[b0 >> 4] * xb0 + table[c0 & 15] * xc0 + table[c0 >> 4] * xd0;
                    int b1 = q1[jj], c1 = q1[jj + 32];
                    d10 += table[b1 & 15] * xa0 + table[b1 >> 4] * xb0 + table[c1 & 15] * xc0 + table[c1 >> 4] * xd0;
                    int b2 = q2[jj], c2 = q2[jj + 32];
                    d20 += table[b2 & 15] * xa0 + table[b2 >> 4] * xb0 + table[c2 & 15] * xc0 + table[c2 >> 4] * xd0;
                    int b3 = q3[jj], c3 = q3[jj + 32];
                    d30 += table[b3 & 15] * xa0 + table[b3 >> 4] * xb0 + table[c3 & 15] * xc0 + table[c3 >> 4] * xd0;
                    int xa1 = x1[jj], xb1 = x1[jj + 32], xc1 = x1[jj + 64], xd1 = x1[jj + 96];
                    b0 = q0[jj]; c0 = q0[jj + 32];
                    d01 += table[b0 & 15] * xa1 + table[b0 >> 4] * xb1 + table[c0 & 15] * xc1 + table[c0 >> 4] * xd1;
                    b1 = q1[jj]; c1 = q1[jj + 32];
                    d11 += table[b1 & 15] * xa1 + table[b1 >> 4] * xb1 + table[c1 & 15] * xc1 + table[c1 >> 4] * xd1;
                    b2 = q2[jj]; c2 = q2[jj + 32];
                    d21 += table[b2 & 15] * xa1 + table[b2 >> 4] * xb1 + table[c2 & 15] * xc1 + table[c2 >> 4] * xd1;
                    b3 = q3[jj]; c3 = q3[jj + 32];
                    d31 += table[b3 & 15] * xa1 + table[b3 >> 4] * xb1 + table[c3 & 15] * xc1 + table[c3 >> 4] * xd1;
                }
                {
                    var jj = j + 2;
                    int xa0 = x0[jj], xb0 = x0[jj + 32], xc0 = x0[jj + 64], xd0 = x0[jj + 96];
                    int b0 = q0[jj], c0 = q0[jj + 32];
                    d00 += table[b0 & 15] * xa0 + table[b0 >> 4] * xb0 + table[c0 & 15] * xc0 + table[c0 >> 4] * xd0;
                    int b1 = q1[jj], c1 = q1[jj + 32];
                    d10 += table[b1 & 15] * xa0 + table[b1 >> 4] * xb0 + table[c1 & 15] * xc0 + table[c1 >> 4] * xd0;
                    int b2 = q2[jj], c2 = q2[jj + 32];
                    d20 += table[b2 & 15] * xa0 + table[b2 >> 4] * xb0 + table[c2 & 15] * xc0 + table[c2 >> 4] * xd0;
                    int b3 = q3[jj], c3 = q3[jj + 32];
                    d30 += table[b3 & 15] * xa0 + table[b3 >> 4] * xb0 + table[c3 & 15] * xc0 + table[c3 >> 4] * xd0;
                    int xa1 = x1[jj], xb1 = x1[jj + 32], xc1 = x1[jj + 64], xd1 = x1[jj + 96];
                    b0 = q0[jj]; c0 = q0[jj + 32];
                    d01 += table[b0 & 15] * xa1 + table[b0 >> 4] * xb1 + table[c0 & 15] * xc1 + table[c0 >> 4] * xd1;
                    b1 = q1[jj]; c1 = q1[jj + 32];
                    d11 += table[b1 & 15] * xa1 + table[b1 >> 4] * xb1 + table[c1 & 15] * xc1 + table[c1 >> 4] * xd1;
                    b2 = q2[jj]; c2 = q2[jj + 32];
                    d21 += table[b2 & 15] * xa1 + table[b2 >> 4] * xb1 + table[c2 & 15] * xc1 + table[c2 >> 4] * xd1;
                    b3 = q3[jj]; c3 = q3[jj + 32];
                    d31 += table[b3 & 15] * xa1 + table[b3 >> 4] * xb1 + table[c3 & 15] * xc1 + table[c3 >> 4] * xd1;
                }
                {
                    var jj = j + 3;
                    int xa0 = x0[jj], xb0 = x0[jj + 32], xc0 = x0[jj + 64], xd0 = x0[jj + 96];
                    int b0 = q0[jj], c0 = q0[jj + 32];
                    d00 += table[b0 & 15] * xa0 + table[b0 >> 4] * xb0 + table[c0 & 15] * xc0 + table[c0 >> 4] * xd0;
                    int b1 = q1[jj], c1 = q1[jj + 32];
                    d10 += table[b1 & 15] * xa0 + table[b1 >> 4] * xb0 + table[c1 & 15] * xc0 + table[c1 >> 4] * xd0;
                    int b2 = q2[jj], c2 = q2[jj + 32];
                    d20 += table[b2 & 15] * xa0 + table[b2 >> 4] * xb0 + table[c2 & 15] * xc0 + table[c2 >> 4] * xd0;
                    int b3 = q3[jj], c3 = q3[jj + 32];
                    d30 += table[b3 & 15] * xa0 + table[b3 >> 4] * xb0 + table[c3 & 15] * xc0 + table[c3 >> 4] * xd0;
                    int xa1 = x1[jj], xb1 = x1[jj + 32], xc1 = x1[jj + 64], xd1 = x1[jj + 96];
                    b0 = q0[jj]; c0 = q0[jj + 32];
                    d01 += table[b0 & 15] * xa1 + table[b0 >> 4] * xb1 + table[c0 & 15] * xc1 + table[c0 >> 4] * xd1;
                    b1 = q1[jj]; c1 = q1[jj + 32];
                    d11 += table[b1 & 15] * xa1 + table[b1 >> 4] * xb1 + table[c1 & 15] * xc1 + table[c1 >> 4] * xd1;
                    b2 = q2[jj]; c2 = q2[jj + 32];
                    d21 += table[b2 & 15] * xa1 + table[b2 >> 4] * xb1 + table[c2 & 15] * xc1 + table[c2 >> 4] * xd1;
                    b3 = q3[jj]; c3 = q3[jj + 32];
                    d31 += table[b3 & 15] * xa1 + table[b3 >> 4] * xb1 + table[c3 & 15] * xc1 + table[c3 >> 4] * xd1;
                }
            }

            acc00 += k00 * d00; bias00 += k00 * oc0;
            acc01 += k01 * d01; bias01 += k01 * oc1;
            acc10 += k10 * d10; bias10 += k10 * oc0;
            acc11 += k11 * d11; bias11 += k11 * oc1;
            acc20 += k20 * d20; bias20 += k20 * oc0;
            acc21 += k21 * d21; bias21 += k21 * oc1;
            acc30 += k30 * d30; bias30 += k30 * oc0;
            acc31 += k31 * d31; bias31 += k31 * oc1;
        }

        y0[n0] = acc00 - bias00;
        y0[n0 + 1] = acc10 - bias10;
        y0[n0 + 2] = acc20 - bias20;
        y0[n0 + 3] = acc30 - bias30;
        y0[ldy + n0] = acc01 - bias01;
        y0[ldy + n0 + 1] = acc11 - bias11;
        y0[ldy + n0 + 2] = acc21 - bias21;
        y0[ldy + n0 + 3] = acc31 - bias31;
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void QuadSingle4(in Cq w, in Activations a, int t, float* y0, int ldy, int n0)
    {
        var groups = w.Groups;
        var table = w.Book->Bytes;
        var byteScale = w.Book->ByteScale;
        var rowBytes = w.RowBytes;
        var cols = a.Cols;
        var stride = a.Stride;
        var v = a.Values + (long)t * cols;
        var sc = a.Scales + (long)t * stride;
        var off = a.Offsets + (long)t * stride;
        var p0 = w.Indices + (long)n0 * rowBytes;
        var p1 = p0 + rowBytes;
        var p2 = p0 + 2 * rowBytes;
        var p3 = p0 + 3 * rowBytes;
        var nr0 = w.Norms + (long)n0 * groups;
        var nr1 = nr0 + groups;
        var nr2 = nr1 + groups;
        var nr3 = nr2 + groups;

        float acc0 = 0, bias0 = 0;
        float acc1 = 0, bias1 = 0;
        float acc2 = 0, bias2 = 0;
        float acc3 = 0, bias3 = 0;

        for (var g = 0; g < groups; g++)
        {
            var s = sc[g];
            var o = off[g];
            var f0 = HalfToSingle(nr0[g]) * byteScale;
            var f1 = HalfToSingle(nr1[g]) * byteScale;
            var f2 = HalfToSingle(nr2[g]) * byteScale;
            var f3 = HalfToSingle(nr3[g]) * byteScale;
            var k0 = f0 * s;
            var k1 = f1 * s;
            var k2 = f2 * s;
            var k3 = f3 * s;

            var x = v + (long)g * Cq.GroupSize;
            var q0 = p0 + (long)g * 64;
            var q1 = p1 + (long)g * 64;
            var q2 = p2 + (long)g * 64;
            var q3 = p3 + (long)g * 64;

            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            for (var j = 0; j < 32; j += 4)
            {
                {
                    var jj = j;
                    int xa = x[jj], xb = x[jj + 32], xc = x[jj + 64], xd = x[jj + 96];
                    int b0 = q0[jj], c0 = q0[jj + 32];
                    d0 += table[b0 & 15] * xa + table[b0 >> 4] * xb + table[c0 & 15] * xc + table[c0 >> 4] * xd;
                    int b1 = q1[jj], c1 = q1[jj + 32];
                    d1 += table[b1 & 15] * xa + table[b1 >> 4] * xb + table[c1 & 15] * xc + table[c1 >> 4] * xd;
                    int b2 = q2[jj], c2 = q2[jj + 32];
                    d2 += table[b2 & 15] * xa + table[b2 >> 4] * xb + table[c2 & 15] * xc + table[c2 >> 4] * xd;
                    int b3 = q3[jj], c3 = q3[jj + 32];
                    d3 += table[b3 & 15] * xa + table[b3 >> 4] * xb + table[c3 & 15] * xc + table[c3 >> 4] * xd;
                }
                {
                    var jj = j + 1;
                    int xa = x[jj], xb = x[jj + 32], xc = x[jj + 64], xd = x[jj + 96];
                    int b0 = q0[jj], c0 = q0[jj + 32];
                    d0 += table[b0 & 15] * xa + table[b0 >> 4] * xb + table[c0 & 15] * xc + table[c0 >> 4] * xd;
                    int b1 = q1[jj], c1 = q1[jj + 32];
                    d1 += table[b1 & 15] * xa + table[b1 >> 4] * xb + table[c1 & 15] * xc + table[c1 >> 4] * xd;
                    int b2 = q2[jj], c2 = q2[jj + 32];
                    d2 += table[b2 & 15] * xa + table[b2 >> 4] * xb + table[c2 & 15] * xc + table[c2 >> 4] * xd;
                    int b3 = q3[jj], c3 = q3[jj + 32];
                    d3 += table[b3 & 15] * xa + table[b3 >> 4] * xb + table[c3 & 15] * xc + table[c3 >> 4] * xd;
                }
                {
                    var jj = j + 2;
                    int xa = x[jj], xb = x[jj + 32], xc = x[jj + 64], xd = x[jj + 96];
                    int b0 = q0[jj], c0 = q0[jj + 32];
                    d0 += table[b0 & 15] * xa + table[b0 >> 4] * xb + table[c0 & 15] * xc + table[c0 >> 4] * xd;
                    int b1 = q1[jj], c1 = q1[jj + 32];
                    d1 += table[b1 & 15] * xa + table[b1 >> 4] * xb + table[c1 & 15] * xc + table[c1 >> 4] * xd;
                    int b2 = q2[jj], c2 = q2[jj + 32];
                    d2 += table[b2 & 15] * xa + table[b2 >> 4] * xb + table[c2 & 15] * xc + table[c2 >> 4] * xd;
                    int b3 = q3[jj], c3 = q3[jj + 32];
                    d3 += table[b3 & 15] * xa + table[b3 >> 4] * xb + table[c3 & 15] * xc + table[c3 >> 4] * xd;
                }
                {
                    var jj = j + 3;
                    int xa = x[jj], xb = x[jj + 32], xc = x[jj + 64], xd = x[jj + 96];
                    int b0 = q0[jj], c0 = q0[jj + 32];
                    d0 += table[b0 & 15] * xa + table[b0 >> 4] * xb + table[c0 & 15] * xc + table[c0 >> 4] * xd;
                    int b1 = q1[jj], c1 = q1[jj + 32];
                    d1 += table[b1 & 15] * xa + table[b1 >> 4] * xb + table[c1 & 15] * xc + table[c1 >> 4] * xd;
                    int b2 = q2[jj], c2 = q2[jj + 32];
                    d2 += table[b2 & 15] * xa + table[b2 >> 4] * xb + table[c2 & 15] * xc + table[c2 >> 4] * xd;
                    int b3 = q3[jj], c3 = q3[jj + 32];
                    d3 += table[b3 & 15] * xa + table[b3 >> 4] * xb + table[c3 & 15] * xc + table[c3 >> 4] * xd;
                }
            }

            acc0 += k0 * d0; bias0 += k0 * o;
            acc1 += k1 * d1; bias1 += k1 * o;
            acc2 += k2 * d2; bias2 += k2 * o;
            acc3 += k3 * d3; bias3 += k3 * o;
        }

        y0[n0] = acc0 - bias0;
        y0[n0 + 1] = acc1 - bias1;
        y0[n0 + 2] = acc2 - bias2;
        y0[n0 + 3] = acc3 - bias3;
    }
}
