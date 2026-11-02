namespace Needle;

// N53: portable register-blocked Kron (scalar, netstandard2.1-safe).
//
// Mirrors W12's KronAvx2/KronAvx512 layout (Blocks.cs): result rows blocked by
// four, two phases through a 4 x cols tile, columns unrolled. The SIMD kernels
// hold 4 rows x 2 vectors of accumulators; here each vector is replaced by
// four scalar accumulators (4 rows x 4 columns = 16 live accumulators), which
// keeps four independent FMA chains per row and hides FMA latency the same
// way. Every z element is loaded once per four k rows (phase 1) and every
// right element once per four k rows (phase 2): a quarter of Scalar.Kron's
// loads, with the same FMA count.
//
// Contract: identical to Scalar.Kron for any non-negative geometry, including
// odd column counts (scalar c tail) and short/ragged inputs. Only a ragged
// row count (resultRows % 4 != 0, which W12's Blocked() gate also rejects)
// delegates to Scalar.Kron.
//
// Wiring (N30 owns the table while it is in flight, so this file installs
// nothing): the future dispatch is one line in HadamardMlp.Kron or the N30
// HadamardMlp entry — PortableMlp.Kron takes exactly Scalar.Kron's arguments.
internal static unsafe class PortableMlp
{
    // Must match HadamardMlp.TileRows: four result rows per tile.
    const int BlockRows = 4;
    // Scalar columns per unroll step: BlockRows x UnrollCols accumulators.
    const int UnrollCols = 4;

    [System.Runtime.CompilerServices.SkipLocalsInit]
    public static void Kron(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
    {
        if (resultRows <= 0 || cols <= 0) return;
        if ((resultRows & 3) != 0)
        {
            // Ragged row count: same fallback W12's Blocked() gate takes.
            Scalar.Kron(z, activeRows, left, right, rows, cols, resultRows, result);
            return;
        }
        // One tile row per k row; every element is written in phase 1 before
        // phase 2 reads it, so skipping zero-init is safe.
        var tile = stackalloc float[BlockRows * cols];
        var mainCols = cols & ~3;
        nint tileStride = cols;
        nint tileStride2 = 2 * cols;
        nint tileStride3 = 3 * cols;
        for (var k = 0; k < resultRows; k += BlockRows)
        {
            // Phase 1: tile[r][c] = sum_i left[i * rows + k + r] * z[i * cols + c].
            for (var c = 0; c < mainCols; c += UnrollCols)
            {
                float a00 = 0, a01 = 0, a02 = 0, a03 = 0;
                float a10 = 0, a11 = 0, a12 = 0, a13 = 0;
                float a20 = 0, a21 = 0, a22 = 0, a23 = 0;
                float a30 = 0, a31 = 0, a32 = 0, a33 = 0;
                for (var i = 0; i < activeRows; i++)
                {
                    var l0 = left[i * rows + k];
                    var l1 = left[i * rows + k + 1];
                    var l2 = left[i * rows + k + 2];
                    var l3 = left[i * rows + k + 3];
                    var z0 = z[i * cols + c];
                    var z1 = z[i * cols + c + 1];
                    var z2 = z[i * cols + c + 2];
                    var z3 = z[i * cols + c + 3];
                    a00 += l0 * z0; a01 += l0 * z1; a02 += l0 * z2; a03 += l0 * z3;
                    a10 += l1 * z0; a11 += l1 * z1; a12 += l1 * z2; a13 += l1 * z3;
                    a20 += l2 * z0; a21 += l2 * z1; a22 += l2 * z2; a23 += l2 * z3;
                    a30 += l3 * z0; a31 += l3 * z1; a32 += l3 * z2; a33 += l3 * z3;
                }
                tile[c] = a00; tile[c + 1] = a01; tile[c + 2] = a02; tile[c + 3] = a03;
                tile[tileStride + c] = a10; tile[tileStride + c + 1] = a11; tile[tileStride + c + 2] = a12; tile[tileStride + c + 3] = a13;
                tile[tileStride2 + c] = a20; tile[tileStride2 + c + 1] = a21; tile[tileStride2 + c + 2] = a22; tile[tileStride2 + c + 3] = a23;
                tile[tileStride3 + c] = a30; tile[tileStride3 + c + 1] = a31; tile[tileStride3 + c + 2] = a32; tile[tileStride3 + c + 3] = a33;
            }
            for (var c = mainCols; c < cols; c++)
            {
                float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (var i = 0; i < activeRows; i++)
                {
                    var w = z[i * cols + c];
                    s0 += left[i * rows + k] * w;
                    s1 += left[i * rows + k + 1] * w;
                    s2 += left[i * rows + k + 2] * w;
                    s3 += left[i * rows + k + 3] * w;
                }
                tile[c] = s0;
                tile[tileStride + c] = s1;
                tile[tileStride2 + c] = s2;
                tile[tileStride3 + c] = s3;
            }
            // Phase 2: result[k + r][c] = sum_j tile[r][j] * right[j * cols + c].
            var output = result + k * cols;
            for (var c = 0; c < mainCols; c += UnrollCols)
            {
                float a00 = 0, a01 = 0, a02 = 0, a03 = 0;
                float a10 = 0, a11 = 0, a12 = 0, a13 = 0;
                float a20 = 0, a21 = 0, a22 = 0, a23 = 0;
                float a30 = 0, a31 = 0, a32 = 0, a33 = 0;
                for (var j = 0; j < cols; j++)
                {
                    var r0 = right[j * cols + c];
                    var r1 = right[j * cols + c + 1];
                    var r2 = right[j * cols + c + 2];
                    var r3 = right[j * cols + c + 3];
                    var t0 = tile[j];
                    var t1 = tile[tileStride + j];
                    var t2 = tile[tileStride2 + j];
                    var t3 = tile[tileStride3 + j];
                    a00 += t0 * r0; a01 += t0 * r1; a02 += t0 * r2; a03 += t0 * r3;
                    a10 += t1 * r0; a11 += t1 * r1; a12 += t1 * r2; a13 += t1 * r3;
                    a20 += t2 * r0; a21 += t2 * r1; a22 += t2 * r2; a23 += t2 * r3;
                    a30 += t3 * r0; a31 += t3 * r1; a32 += t3 * r2; a33 += t3 * r3;
                }
                output[c] = a00; output[c + 1] = a01; output[c + 2] = a02; output[c + 3] = a03;
                output[tileStride + c] = a10; output[tileStride + c + 1] = a11; output[tileStride + c + 2] = a12; output[tileStride + c + 3] = a13;
                output[tileStride2 + c] = a20; output[tileStride2 + c + 1] = a21; output[tileStride2 + c + 2] = a22; output[tileStride2 + c + 3] = a23;
                output[tileStride3 + c] = a30; output[tileStride3 + c + 1] = a31; output[tileStride3 + c + 2] = a32; output[tileStride3 + c + 3] = a33;
            }
            for (var c = mainCols; c < cols; c++)
            {
                float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (var j = 0; j < cols; j++)
                {
                    var w = right[j * cols + c];
                    s0 += tile[j] * w;
                    s1 += tile[tileStride + j] * w;
                    s2 += tile[tileStride2 + j] * w;
                    s3 += tile[tileStride3 + j] * w;
                }
                output[c] = s0;
                output[tileStride + c] = s1;
                output[tileStride2 + c] = s2;
                output[tileStride3 + c] = s3;
            }
        }
    }
}
