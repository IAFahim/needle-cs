namespace Needle;

// N52: LUT-GEMV (T-MAC style) integer matmul for the portable provider.
//
// Per (token, 128-group) the kernel precomputes byte-indexed tables over the
// activations and the weight codebook, turning every packed weight byte into
// one load plus one add instead of four multiply-adds, shifts and masks:
//
//   2-bit: T[j][b] = book[(b>>0)&3]*x[j] + book[(b>>2)&3]*x[32+j]
//                  + book[(b>>4)&3]*x[64+j] + book[(b>>6)&3]*x[96+j]
//          j in 0..32, b in 0..256: 32*256 int32 = 32 KiB per group.
//   4-bit: one 32x256 table per nibble half h in {0,1}:
//          Th[j][b] = book[b&15]*x[2h*32+j] + book[b>>4]*x[(2h+1)*32+j]
//          (64 KiB per group: lo half then hi half).
//
// Integer sums are bit-exact with Scalar.MatMulRows by construction: the dots
// cannot overflow int32 (about 2M worst case), so every summation order of the
// same 128 products per (row, group) is identical, and the per-entry sums
// below add the same four (2-bit) or two (4-bit) products Scalar adds. The
// float fold keeps Scalar's exact order per (row, group):
// scaled = HalfToSingle(norm) * byteScale * scale,
// accumulator += scaled * dot and bias += scaled * offset in group order,
// finishing with y = accumulator - bias. Row tails (< 4 rows) run Scalar
// itself, so they are exact trivially.
//
// Table build cost is paid once per (token, group) and amortized over rows:
// the build first forms the plane level products (2-bit: 4 planes x 4 levels;
// 4-bit: 2 x 16 levels) and then fills entries with one add each, storing
// sequentially with no per-entry masking (2-bit ~512 IMUL + 26k adds per
// group). This is aimed at the N51 finding (IMUL-throughput-bound): the dot
// loop itself does no multiplies at all.
//
// Loop order is groups-outer, rows-inner everywhere below: one group's table
// (32 KiB) stays L1-resident across the whole row slice, while a groups-inner
// (N51-style) order would cycle groups*32 KiB through L1 per row quad. The
// float accumulators therefore live in small stack arrays sized by SliceRows.
//
// Sharing: the tables depend only on (activations, codebook bytes). The model
// keeps one global codebook per bit-width (Model.cs Reader: every Cq points
// at book2 or book4), so Q, K, V and Gate matrices over the same Activations
// share one build. BuildTables writes the tables once into caller-owned
// scratch; MatMulRowsWithTables consumes them for any matrix with the same
// book. The standalone MatMulRows drop-in below owns no scratch: it rebuilds
// per 1024-row slice (32 KiB table + 8 KiB accumulators on the stack, reused
// per group) and delegates small row counts and T != 1 to PortableMatMul.
//
// Contract (plan section 2 provider rules): pure functions, no heap
// allocation (only stack scratch in the slice paths), pointer + scalar
// arguments only, safe from any thread as long as callers give each thread
// disjoint outputs/scratch.
//
// Wiring API (for the N30 owner; this file adds no dispatch itself):
//   PortableLut.MatMulRows(in Cq, in Activations, int tokens, float* y,
//     int ldy, int rowStart, int rowEnd) — a drop-in for Scalar.MatMulRows.
//   int ludInts = PortableLut.TableInts(w.Groups, w.Bits) — scratch ints per
//     token for the shared path (e.g. DimAct 768 -> 6 groups: 48K/96K ints).
//   PortableLut.BuildTables(w, a, token, tables) — once per token per
//     bit-width; tables must be TableInts(w.Groups, w.Bits) ints.
//   PortableLut.MatMulRowsWithTables(w, a, token, tables, y, ldy, rowStart,
//     rowEnd) — per matrix consuming the same (a, token, book); y is the base
//     output pointer (the token offset is applied inside).
//   Suggested dispatch in Quant.MatMulRows, after the table check and before
//   the Scalar fallback (or wherever the portable provider slots in):
//     if (!Simd.Enabled) { PortableLut.MatMulRows(w, a, tokens, y, ldy, rowStart, rowEnd); return; }
internal static unsafe class PortableLut
{
    public const int Lanes = 32;
    public const int Values = 256;
    public const int TableInts2 = 32 * 256; // per group, 2-bit (32 KiB)
    public const int TableInts4 = 2 * 32 * 256; // per group, 4-bit (64 KiB)
    public static int TableInts(int groups, int bits) => groups * (bits == 2 ? TableInts2 : TableInts4);

    // Below these row counts the table build exceeds the lookup savings; the
    // drop-in delegates to N51 (still bit-exact). N52 bench at T=1:
    // 2-bit standalone wins from ~192 rows (CoreCLR+ns2.1 2.4x, Mono 1.4x)
    // up to 3.5-4.2x at 768+ rows; below 192 Mono loses (0.85x at 96 rows).
    // 4-bit pays a 2x bigger build and 2x the dot loads, so it wins only from
    // ~576 rows (1.1-1.3x standalone both runtimes; dot-only 1.6-1.9x).
    // The shared path has no per-matrix build, so the N30 owner may call
    // WithTables below these cutoffs whenever the tables already exist for a
    // sibling matrix (dot-only wins from the smallest shapes: 4.3x/3.3x at
    // 96 rows, 2-bit, CoreCLR/Mono).
    const int MinRowsForLut2 = 192;
    const int MinRowsForLut4 = 576;

    // Row slice for the groups-outer order: accumulators (acc+bias, plus the
    // 4-bit half-dot staging) live on the stack, the 32 KiB table scratch is
    // reused per group, so the stack peak stays ~44 KiB regardless of matrix
    // height. Larger slices rebuild fewer tables; 1024 keeps the peak modest
    // for small-stack runtimes (IL2CPP workers, Mono threads).
    const int SliceRows = 1024;

    public static void MatMulRows(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    {
        if (tokens <= 0 || rowEnd <= rowStart) return;
        if (w.Bits == 2)
        {
            if (tokens != 1 || rowEnd - rowStart < MinRowsForLut2)
            {
                PortableMatMul.MatMulRows(w, a, tokens, y, ldy, rowStart, rowEnd);
                return;
            }
            Standalone2(w, a, y, ldy, rowStart, rowEnd);
        }
        else
        {
            if (tokens != 1 || rowEnd - rowStart < MinRowsForLut4)
            {
                PortableMatMul.MatMulRows(w, a, tokens, y, ldy, rowStart, rowEnd);
                return;
            }
            Standalone4(w, a, y, ldy, rowStart, rowEnd);
        }
    }

    public static void BuildTables(in Cq w, in Activations a, int token, int* tables)
    {
        var book = w.Book->Bytes;
        var x = a.Values + (long)token * a.Cols;
        if (w.Bits == 2)
        {
            for (var g = 0; g < w.Groups; g++)
                BuildGroup2(book, x + (long)g * Cq.GroupSize, tables + (long)g * TableInts2);
        }
        else
        {
            for (var g = 0; g < w.Groups; g++)
            {
                var xg = x + (long)g * Cq.GroupSize;
                var tg = tables + (long)g * TableInts4;
                BuildHalf4(book, xg, xg + 32, tg);
                BuildHalf4(book, xg + 64, xg + 96, tg + TableInts2);
            }
        }
    }

    public static void MatMulRowsWithTables(in Cq w, in Activations a, int token, int* tables, float* y, int ldy, int rowStart, int rowEnd)
    {
        if (rowEnd <= rowStart) return;
        var y0 = y + (long)token * ldy;
        var scales = a.Scales + (long)token * a.Stride;
        var offsets = a.Offsets + (long)token * a.Stride;
        var quadEnd = rowStart + ((rowEnd - rowStart) & ~3);
        for (var r0 = rowStart; r0 < quadEnd; r0 += SliceRows)
        {
            var count = quadEnd - r0 < SliceRows ? quadEnd - r0 : SliceRows;
            if (w.Bits == 2) SliceWithTables2(w, scales, offsets, tables, y0, r0, count);
            else SliceWithTables4(w, scales, offsets, tables, y0, r0, count);
        }
        if (quadEnd < rowEnd)
            Scalar.MatMulRows(w, a.Slice(token), 1, y0, ldy, quadEnd, rowEnd);
    }

    // Bit-identical copy of Scalar.HalfToSingle, forced inline: the group loop
    // calls this per (row, group) and the JIT leaves cross-class calls
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

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void BuildGroup2(byte* book, sbyte* x, int* t)
    {
        var b0 = book[0];
        var b1 = book[1];
        var b2 = book[2];
        var b3 = book[3];
        // Level products per plane, then the 16 low-pair sums: entry
        // (c0|c1<<2|c2<<4|c3<<6) = lo[c0|c1<<2] + p[8+c2] + p[12+c3].
        // The inner store runs sequentially over 16 contiguous entries with
        // no per-entry masking; the 16 lo[] sums are shared by all 16 (c2,c3).
        int* p = stackalloc int[16];
        int* lo = stackalloc int[16];
        for (var j = 0; j < Lanes; j++)
        {
            int xa = x[j], xb = x[j + 32], xc = x[j + 64], xd = x[j + 96];
            p[0] = b0 * xa; p[1] = b1 * xa; p[2] = b2 * xa; p[3] = b3 * xa;
            p[4] = b0 * xb; p[5] = b1 * xb; p[6] = b2 * xb; p[7] = b3 * xb;
            p[8] = b0 * xc; p[9] = b1 * xc; p[10] = b2 * xc; p[11] = b3 * xc;
            p[12] = b0 * xd; p[13] = b1 * xd; p[14] = b2 * xd; p[15] = b3 * xd;
            for (var k = 0; k < 16; k++) lo[k] = p[k & 3] + p[4 + ((k >> 2) & 3)];
            var tj = t + j * Values;
            for (var hi = 0; hi < 16; hi++)
            {
                var v = p[8 + (hi & 3)] + p[12 + (hi >> 2)];
                var dst = tj + hi * 16;
                for (var k = 0; k < 16; k++) dst[k] = lo[k] + v;
            }
        }
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void BuildHalf4(byte* book, sbyte* xa, sbyte* xb, int* t)
    {
        int* p = stackalloc int[32];
        for (var j = 0; j < Lanes; j++)
        {
            int va = xa[j], vb = xb[j];
            for (var c = 0; c < 16; c++)
            {
                p[c] = book[c] * va;
                p[16 + c] = book[c] * vb;
            }
            // Entry (lo|hi<<4) = p[lo] + p[16+hi]: hoist the hi term, store
            // 16 contiguous entries per hi with no per-entry masking.
            var tj = t + j * Values;
            for (var hi = 0; hi < 16; hi++)
            {
                var v = p[16 + hi];
                var dst = tj + hi * 16;
                for (var k = 0; k < 16; k++) dst[k] = p[k] + v;
            }
        }
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void SliceWithTables2(in Cq w, float* scales, float* offsets, int* tables, float* y0, int r0, int count)
    {
        var groups = w.Groups;
        var byteScale = w.Book->ByteScale;
        var rowBytes = w.RowBytes;
        float* acc = stackalloc float[SliceRows];
        float* biz = stackalloc float[SliceRows];
        for (var i = 0; i < count; i++) { acc[i] = 0; biz[i] = 0; }
        var prow = w.Indices + (long)r0 * rowBytes;
        var pnorm = w.Norms + (long)r0 * groups;
        for (var g = 0; g < groups; g++)
        {
            var s = scales[g];
            var o = offsets[g];
            var T = tables + (long)g * TableInts2;
            for (var i = 0; i < count; i += 4)
            {
                var q0 = prow + (long)i * rowBytes + (long)g * 32;
                var q1 = q0 + rowBytes;
                var q2 = q0 + 2 * rowBytes;
                var q3 = q0 + 3 * rowBytes;
                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                for (var j = 0; j < Lanes; j += 4)
                {
                    var t0 = T + (j << 8);
                    var t1 = T + ((j + 1) << 8);
                    var t2 = T + ((j + 2) << 8);
                    var t3 = T + ((j + 3) << 8);
                    d0 += t0[q0[j]] + t1[q0[j + 1]] + t2[q0[j + 2]] + t3[q0[j + 3]];
                    d1 += t0[q1[j]] + t1[q1[j + 1]] + t2[q1[j + 2]] + t3[q1[j + 3]];
                    d2 += t0[q2[j]] + t1[q2[j + 1]] + t2[q2[j + 2]] + t3[q2[j + 3]];
                    d3 += t0[q3[j]] + t1[q3[j + 1]] + t2[q3[j + 2]] + t3[q3[j + 3]];
                }
                var f0 = HalfToSingle(pnorm[(long)i * groups + g]) * byteScale;
                var f1 = HalfToSingle(pnorm[(long)(i + 1) * groups + g]) * byteScale;
                var f2 = HalfToSingle(pnorm[(long)(i + 2) * groups + g]) * byteScale;
                var f3 = HalfToSingle(pnorm[(long)(i + 3) * groups + g]) * byteScale;
                var k0 = f0 * s;
                var k1 = f1 * s;
                var k2 = f2 * s;
                var k3 = f3 * s;
                acc[i] += k0 * d0; biz[i] += k0 * o;
                acc[i + 1] += k1 * d1; biz[i + 1] += k1 * o;
                acc[i + 2] += k2 * d2; biz[i + 2] += k2 * o;
                acc[i + 3] += k3 * d3; biz[i + 3] += k3 * o;
            }
        }
        for (var i = 0; i < count; i++) y0[r0 + i] = acc[i] - biz[i];
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void SliceWithTables4(in Cq w, float* scales, float* offsets, int* tables, float* y0, int r0, int count)
    {
        var groups = w.Groups;
        var byteScale = w.Book->ByteScale;
        var rowBytes = w.RowBytes;
        float* acc = stackalloc float[SliceRows];
        float* biz = stackalloc float[SliceRows];
        for (var i = 0; i < count; i++) { acc[i] = 0; biz[i] = 0; }
        var prow = w.Indices + (long)r0 * rowBytes;
        var pnorm = w.Norms + (long)r0 * groups;
        for (var g = 0; g < groups; g++)
        {
            var s = scales[g];
            var o = offsets[g];
            var T = tables + (long)g * TableInts4;
            var U = T + TableInts2;
            for (var i = 0; i < count; i += 4)
            {
                var q0 = prow + (long)i * rowBytes + (long)g * 64;
                var q1 = q0 + rowBytes;
                var q2 = q0 + 2 * rowBytes;
                var q3 = q0 + 3 * rowBytes;
                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                for (var j = 0; j < Lanes; j += 4)
                {
                    var t0 = T + (j << 8);
                    var t1 = T + ((j + 1) << 8);
                    var t2 = T + ((j + 2) << 8);
                    var t3 = T + ((j + 3) << 8);
                    var u0 = U + (j << 8);
                    var u1 = U + ((j + 1) << 8);
                    var u2 = U + ((j + 2) << 8);
                    var u3 = U + ((j + 3) << 8);
                    d0 += t0[q0[j]] + t1[q0[j + 1]] + t2[q0[j + 2]] + t3[q0[j + 3]]
                        + u0[q0[j + 32]] + u1[q0[j + 33]] + u2[q0[j + 34]] + u3[q0[j + 35]];
                    d1 += t0[q1[j]] + t1[q1[j + 1]] + t2[q1[j + 2]] + t3[q1[j + 3]]
                        + u0[q1[j + 32]] + u1[q1[j + 33]] + u2[q1[j + 34]] + u3[q1[j + 35]];
                    d2 += t0[q2[j]] + t1[q2[j + 1]] + t2[q2[j + 2]] + t3[q2[j + 3]]
                        + u0[q2[j + 32]] + u1[q2[j + 33]] + u2[q2[j + 34]] + u3[q2[j + 35]];
                    d3 += t0[q3[j]] + t1[q3[j + 1]] + t2[q3[j + 2]] + t3[q3[j + 3]]
                        + u0[q3[j + 32]] + u1[q3[j + 33]] + u2[q3[j + 34]] + u3[q3[j + 35]];
                }
                var f0 = HalfToSingle(pnorm[(long)i * groups + g]) * byteScale;
                var f1 = HalfToSingle(pnorm[(long)(i + 1) * groups + g]) * byteScale;
                var f2 = HalfToSingle(pnorm[(long)(i + 2) * groups + g]) * byteScale;
                var f3 = HalfToSingle(pnorm[(long)(i + 3) * groups + g]) * byteScale;
                var k0 = f0 * s;
                var k1 = f1 * s;
                var k2 = f2 * s;
                var k3 = f3 * s;
                acc[i] += k0 * d0; biz[i] += k0 * o;
                acc[i + 1] += k1 * d1; biz[i + 1] += k1 * o;
                acc[i + 2] += k2 * d2; biz[i + 2] += k2 * o;
                acc[i + 3] += k3 * d3; biz[i + 3] += k3 * o;
            }
        }
        for (var i = 0; i < count; i++) y0[r0 + i] = acc[i] - biz[i];
    }

    // Standalone path: no caller scratch. Rows run in slices; per (slice,
    // group) the table is rebuilt into 32 KiB of stack scratch and stays
    // L1-resident across the slice's quads.
    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void Standalone2(in Cq w, in Activations a, float* y, int ldy, int rowStart, int rowEnd)
    {
        var values = a.Values;
        var scales = a.Scales;
        var offsets = a.Offsets;
        var book = w.Book->Bytes;

        int* lut = stackalloc int[TableInts2];
        float* acc = stackalloc float[SliceRows];
        float* biz = stackalloc float[SliceRows];

        for (var r0 = rowStart; r0 < rowEnd; r0 += SliceRows)
        {
            var count = rowEnd - r0 < SliceRows ? rowEnd - r0 : SliceRows;
            var quad = count & ~3;
            for (var i = 0; i < quad; i++) { acc[i] = 0; biz[i] = 0; }
            SliceStandalone2(w, book, values, scales, offsets, lut, acc, biz, r0, quad);
            for (var i = 0; i < quad; i++) y[r0 + i] = acc[i] - biz[i];
            if (quad < count)
                Scalar.MatMulRows(w, a, 1, y, ldy, r0 + quad, r0 + count);
        }
    }

    static void SliceStandalone2(in Cq w, byte* book, sbyte* values, float* scales, float* offsets, int* lut, float* acc, float* biz, int r0, int quad)
    {
        var groups = w.Groups;
        var byteScale = w.Book->ByteScale;
        var rowBytes = w.RowBytes;
        var prow = w.Indices + (long)r0 * rowBytes;
        var pnorm = w.Norms + (long)r0 * groups;
        for (var g = 0; g < groups; g++)
        {
            BuildGroup2(book, values + (long)g * Cq.GroupSize, lut);
            var s = scales[g];
            var o = offsets[g];
            for (var i = 0; i < quad; i += 4)
            {
                var q0 = prow + (long)i * rowBytes + (long)g * 32;
                var q1 = q0 + rowBytes;
                var q2 = q0 + 2 * rowBytes;
                var q3 = q0 + 3 * rowBytes;
                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                for (var j = 0; j < Lanes; j += 4)
                {
                    var t0 = lut + (j << 8);
                    var t1 = lut + ((j + 1) << 8);
                    var t2 = lut + ((j + 2) << 8);
                    var t3 = lut + ((j + 3) << 8);
                    d0 += t0[q0[j]] + t1[q0[j + 1]] + t2[q0[j + 2]] + t3[q0[j + 3]];
                    d1 += t0[q1[j]] + t1[q1[j + 1]] + t2[q1[j + 2]] + t3[q1[j + 3]];
                    d2 += t0[q2[j]] + t1[q2[j + 1]] + t2[q2[j + 2]] + t3[q2[j + 3]];
                    d3 += t0[q3[j]] + t1[q3[j + 1]] + t2[q3[j + 2]] + t3[q3[j + 3]];
                }
                var f0 = HalfToSingle(pnorm[(long)i * groups + g]) * byteScale;
                var f1 = HalfToSingle(pnorm[(long)(i + 1) * groups + g]) * byteScale;
                var f2 = HalfToSingle(pnorm[(long)(i + 2) * groups + g]) * byteScale;
                var f3 = HalfToSingle(pnorm[(long)(i + 3) * groups + g]) * byteScale;
                var k0 = f0 * s;
                var k1 = f1 * s;
                var k2 = f2 * s;
                var k3 = f3 * s;
                acc[i] += k0 * d0; biz[i] += k0 * o;
                acc[i + 1] += k1 * d1; biz[i + 1] += k1 * o;
                acc[i + 2] += k2 * d2; biz[i + 2] += k2 * o;
                acc[i + 3] += k3 * d3; biz[i + 3] += k3 * o;
            }
        }
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void Standalone4(in Cq w, in Activations a, float* y, int ldy, int rowStart, int rowEnd)
    {
        var values = a.Values;
        var scales = a.Scales;
        var offsets = a.Offsets;
        var book = w.Book->Bytes;

        int* lut = stackalloc int[TableInts2];
        int* e = stackalloc int[SliceRows];
        float* acc = stackalloc float[SliceRows];
        float* biz = stackalloc float[SliceRows];

        for (var r0 = rowStart; r0 < rowEnd; r0 += SliceRows)
        {
            var count = rowEnd - r0 < SliceRows ? rowEnd - r0 : SliceRows;
            var quad = count & ~3;
            for (var i = 0; i < quad; i++) { acc[i] = 0; biz[i] = 0; }
            SliceStandalone4(w, book, values, scales, offsets, lut, e, acc, biz, r0, quad);
            for (var i = 0; i < quad; i++) y[r0 + i] = acc[i] - biz[i];
            if (quad < count)
                Scalar.MatMulRows(w, a, 1, y, ldy, r0 + quad, r0 + count);
        }
    }

    static void SliceStandalone4(in Cq w, byte* book, sbyte* values, float* scales, float* offsets, int* lut, int* e, float* acc, float* biz, int r0, int quad)
    {
        var groups = w.Groups;
        var byteScale = w.Book->ByteScale;
        var rowBytes = w.RowBytes;
        var prow = w.Indices + (long)r0 * rowBytes;
        var pnorm = w.Norms + (long)r0 * groups;
        for (var g = 0; g < groups; g++)
        {
            var xg = values + (long)g * Cq.GroupSize;
            var s = scales[g];
            var o = offsets[g];
            BuildHalf4(book, xg, xg + 32, lut);
            for (var i = 0; i < quad; i += 4)
            {
                var q0 = prow + (long)i * rowBytes + (long)g * 64;
                var q1 = q0 + rowBytes;
                var q2 = q0 + 2 * rowBytes;
                var q3 = q0 + 3 * rowBytes;
                e[i] = DotHalf(lut, q0);
                e[i + 1] = DotHalf(lut, q1);
                e[i + 2] = DotHalf(lut, q2);
                e[i + 3] = DotHalf(lut, q3);
            }
            BuildHalf4(book, xg + 64, xg + 96, lut);
            for (var i = 0; i < quad; i += 4)
            {
                var q0 = prow + (long)i * rowBytes + (long)g * 64 + 32;
                var q1 = q0 + rowBytes;
                var q2 = q0 + 2 * rowBytes;
                var q3 = q0 + 3 * rowBytes;
                var f0 = HalfToSingle(pnorm[(long)i * groups + g]) * byteScale;
                var f1 = HalfToSingle(pnorm[(long)(i + 1) * groups + g]) * byteScale;
                var f2 = HalfToSingle(pnorm[(long)(i + 2) * groups + g]) * byteScale;
                var f3 = HalfToSingle(pnorm[(long)(i + 3) * groups + g]) * byteScale;
                var k0 = f0 * s;
                var k1 = f1 * s;
                var k2 = f2 * s;
                var k3 = f3 * s;
                var d0 = e[i] + DotHalf(lut, q0);
                var d1 = e[i + 1] + DotHalf(lut, q1);
                var d2 = e[i + 2] + DotHalf(lut, q2);
                var d3 = e[i + 3] + DotHalf(lut, q3);
                acc[i] += k0 * d0; biz[i] += k0 * o;
                acc[i + 1] += k1 * d1; biz[i + 1] += k1 * o;
                acc[i + 2] += k2 * d2; biz[i + 2] += k2 * o;
                acc[i + 3] += k3 * d3; biz[i + 3] += k3 * o;
            }
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    static int DotHalf(int* lut, byte* q)
    {
        var d = 0;
        for (var j = 0; j < Lanes; j += 4)
        {
            var t0 = lut + (j << 8);
            var t1 = lut + ((j + 1) << 8);
            var t2 = lut + ((j + 2) << 8);
            var t3 = lut + ((j + 3) << 8);
            d += t0[q[j]] + t1[q[j + 1]] + t2[q[j + 2]] + t3[q[j + 3]];
        }
        return d;
    }
}
