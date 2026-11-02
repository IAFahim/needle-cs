// N40: Phase-2 fused layer member (Track B).
//
// Ports EngineCore<TClock>.ExecuteLayer and its stage functions
// (HyperNorm, HyperPhi, HyperMix, EngramGather, EngramMatMul, EngramBlend,
// AttentionPrepare, AttentionQkvg, TapsHistory, AttentionHeads,
// AttentionCore, AttentionGate, AttentionOut, PostNorm, Mlp, Combine) as ONE
// Burst entry taking a LayerJob mirror pointer plus a member index. Barriers
// use Interlocked plus a Common.Pause spin loop; no managed calls inside.
//
// VERDICT (acceptance: >= 10% decode win over the per-kernel table, else
// close with numbers): NOT WORTH wiring in. Measured headroom is ~0.1-0.4%,
// about 75x below the bar. Numbers (all on the N40 probe box / repo evidence):
//   - N12-after CoreCLR T4 decode: 17,623 us/token best (ref/bench/
//     n12-after-coreclr.json), 20 layers => ~881 us/token/layer.
//   - Function-pointer transition: 1.7 ns direct calli (probe, CoreCLR x64,
//     20M iterations); H5 budgets < 100 ns for a managed->Burst call.
//     Per member per layer the fused entry removes at most ~38 calls
//     (member owning the decode token: 2 HyperNorm + 1 HyperPhi + 2 HyperMix
//     + 2 AttentionPrepare + ~4 Qkvg + 3 Taps + ~12 Heads + 1 Core + 2 Gate
//     + ~3 AttnOut + 1 PostNorm + 1 Mlp + 4 Combine) and ~15 on average.
//     38 x 30 ns ~= 1.1 us/layer => ~23 us/token => 0.13% of decode.
//     Even at the 100 ns H5 ceiling: 0.4%.
//   - Barriers stay: ExecuteLayer issues 11 barriers per layer (14 on the 5
//     engram layers; TapsHistory and AttentionCore add internal ones), all
//     required by stage data dependencies. Hot rendezvous is 0.44 us (V12),
//     so ~5 us/layer of synchronization cannot fuse away.
//   - Fundamentals: decode is weight-bandwidth-bound (~17.5 MB/token across
//     the same matrices); fusion moves identical bytes and FLOPs, and
//     RunLayerTeam already issues one team dispatch per layer, so no
//     dispatch collapses either.
//   - Restructuring barriers away (redundant history loads, sharded arrivals
//     zeroing) could save at most ~3 x 0.44 us x 20 ~= 26 us/token (0.15%)
//     and changes engine scheduling invariants, so it belongs to the engine
//     owner, not to this file.
// The file is kept: it is the complete, exact port, so the finding above is
// checkable instruction by instruction, and it becomes useful the day some
// future geometry makes call overhead matter.
//
// Layout contract (mirrors use explicit offsets, not declaration order):
// every offset below was read off the built net10 assembly with a reflection
// probe (Marshal.OffsetOf per field, Unsafe.SizeOf per struct) on 2026-11-03:
//   Geometry 120, Cq 40, Codebook 100, Activations 40, MlpWeights 104,
//   LayerWeights 376, HyperConnections 184, EngramSite 128, TeamBarrier 16,
//   MlpBatch 56, AttentionBatch/Tile 128. LayerJob (EngineCore<NoClock>)
//   sums to 1248 bytes: Barrier @0, NoClock clock byte @8, Members @12 (+6
//   ints), Geometry @40, Weights @160, Hyper @536, Engram @720,
//   StreamAct @848, DimAct @888, then 40 pointers @928..1248.
// The engine side is frozen by a single [StructLayout(Sequential)] line on
// LayerJob (N40's one Engine.cs line); the numeric half of the freeze is
// JobSize + AssertLayout below, which N30's owner (or a core test) calls as
// BurstLayer.AssertLayout(Unsafe.SizeOf<LayerJob>()) from inside the
// assembly where LayerJob is visible. If TClock ever stops being NoClock,
// the size moves and the assert fires instead of silently misreading.
//
// Burst API provenance (names verified, not guessed):
//   - Unity.Burst.Intrinsics.Common.Pause: cross-platform pause, documented
//     under Unity 7000.0 scripting reference (Common/Pause).
//   - System.Threading.Interlocked: Burst supports all Interlocked methods
//     (7000.0 "C#/System namespace support" page); arrivals/participants
//     are 4-byte aligned inside the 16-byte arena barrier, satisfying the
//     natural-alignment rule on every 64-bit target.
//   - System.Threading.Volatile: non-generic Read/Write supported (same page).
//   - No managed calls inside Invoke: only sibling Burst entries, MathF,
//     Interlocked/Volatile/Common, pointer arithmetic, stackalloc.
//
// Telemetry caveat: the fused entry cannot call IStageClock (a managed
// interface call). Stage laps are skipped; RunLayerTeam's clock.PoolRun and
// the lap-free path behave like a NoClock run. An N12 stage profile therefore
// cannot be collected through this entry; end-to-end decode us/token still
// compares directly.
//
// Wiring note: KernelTable has no LayerMember field yet (N30 reserved it).
// Install() compiles Invoke and stashes the raw pointer in Function without
// touching the table; N30's owner adds the field plus the one-line store and
// the Engine.cs RunLayerTeam branch. Install is idempotent and never
// overwrites.
//
// C# 9 safe: block namespace, no target-typed new, no static lambdas.
using System;
using System.Runtime.InteropServices;
using System.Threading;
using Unity.Burst;
using Unity.Burst.Intrinsics;

namespace Needle.BurstKernels
{
    public static unsafe class BurstLayer
    {
        // ---- constants mirrored from the engine (Engine.cs / Blocks.cs) ----
        const int Lanes = 4;
        const int ConvTaps = 3;
        const int EngramHistory = 9;
        const int EngramTaps = 4;
        const int MaxTileHeads = 6;
        const int PartialFloats = MaxTileHeads * 64 + 16; // 400, matches Attention.PartialFloats
        const int PeaksOffset = MaxTileHeads * 64;        // 384
        const int SumsOffset = MaxTileHeads * 64 + 8;     // 392
        const int MinChunkPositions = 48;
        const int MaxChunks = 8;
        const int TasksPerThread = 3;
        const int MinimumRowsPerTask = 8;
        const int RowBlock = 4; // Quant.RowBlock

        // ---- explicit-layout mirrors (offsets: reflection probe, see header) ----

        [StructLayout(LayoutKind.Explicit, Size = 16)]
        public struct LayerBarrier
        {
            [FieldOffset(0)] public long Phase;
            [FieldOffset(8)] public int Arrived;
            [FieldOffset(12)] public int Participants;
        }

        [StructLayout(LayoutKind.Explicit, Size = 40)]
        public struct LayerCq
        {
            [FieldOffset(0)] public byte* Indices;
            [FieldOffset(8)] public ushort* Norms;
            // Codebook base as bytes: Levels[16] floats @+0, Bytes[32] @+64,
            // ByteScale float @+96 (Codebook size 100). Scalar lane reads the
            // table/scale by offset; GatherRow reinterprets via BurstMisc.
            [FieldOffset(16)] public byte* Book;
            [FieldOffset(24)] public int Rows;
            [FieldOffset(28)] public int Cols;
            [FieldOffset(32)] public int Bits;
        }

        [StructLayout(LayoutKind.Explicit, Size = 40)]
        public struct LayerActivations
        {
            [FieldOffset(0)] public sbyte* Values;
            [FieldOffset(8)] public float* Scales;
            [FieldOffset(16)] public float* Offsets;
            [FieldOffset(24)] public int Cols;
            [FieldOffset(28)] public int Groups;
            [FieldOffset(32)] public int Stride;
        }

        [StructLayout(LayoutKind.Explicit, Size = 120)]
        public struct LayerGeometry
        {
            [FieldOffset(0)] public int Vocab;
            [FieldOffset(4)] public int Dim;
            [FieldOffset(8)] public int Heads;
            [FieldOffset(12)] public int KvHeads;
            [FieldOffset(16)] public int Layers;
            [FieldOffset(20)] public int QkDim;
            [FieldOffset(24)] public int VDim;
            [FieldOffset(28)] public int MaxSeq;
            [FieldOffset(32)] public int HadaN;
            [FieldOffset(36)] public int Lanes;
            [FieldOffset(40)] public int SlidingWindow;
            [FieldOffset(44)] public int ConvTaps;
            [FieldOffset(48)] public int EngramSlots;
            [FieldOffset(52)] public int EngramSub;
            [FieldOffset(56)] public int EngramTables;
            [FieldOffset(60)] public int EngramConvTaps;
            [FieldOffset(64)] public int EngramDilation;
            [FieldOffset(68)] public int EngramSeedHeads;
            [FieldOffset(72)] public int KvWindow;
            [FieldOffset(76)] public int OrderCount;
            [FieldOffset(80)] public int SiteCount;
            [FieldOffset(88)] public ulong GlobalMask;
            [FieldOffset(96)] public float RopeTheta;
            [FieldOffset(104)] public int* Orders;
            [FieldOffset(112)] public int* Sites;
            // Computed, exactly like Geometry (not fields, so layout is untouched).
            public int QueryWidth => Heads * QkDim;
            public int KeyWidth => KvHeads * QkDim;
            public int ValueWidth => KvHeads * VDim;
            public int OutputWidth => Heads * VDim;
        }

        [StructLayout(LayoutKind.Explicit, Size = 376)]
        public struct LayerWeightsMirror
        {
            [FieldOffset(0)] public float* NormIn;
            [FieldOffset(8)] public LayerCq Q;
            [FieldOffset(48)] public LayerCq K;
            [FieldOffset(88)] public LayerCq V;
            [FieldOffset(128)] public float* QTaps;
            [FieldOffset(136)] public float* KTaps;
            [FieldOffset(144)] public float* VTaps;
            [FieldOffset(152)] public float* QNorm;
            [FieldOffset(160)] public float* KNorm;
            [FieldOffset(168)] public LayerCq Gate;
            [FieldOffset(208)] public LayerCq Out;
            [FieldOffset(248)] public float* PostNorm;
            [FieldOffset(256)] public float AttnGate;
            [FieldOffset(264)] public float* PreHada;
            // MlpWeights is 13 float* (104 bytes, probe); BurstMlpWeights is
            // its identical twin, so the engine pointer is used directly.
            [FieldOffset(272)] public BurstMlpWeights* Mlp;
        }

        [StructLayout(LayoutKind.Explicit, Size = 184)]
        public struct LayerHyper
        {
            [FieldOffset(0)] public float* APre;
            [FieldOffset(8)] public float* APost;
            [FieldOffset(16)] public float* ARes;
            [FieldOffset(24)] public float* BPre;
            [FieldOffset(32)] public float* BPost;
            [FieldOffset(40)] public float* BRes;
            [FieldOffset(48)] public LayerCq PhiPre;
            [FieldOffset(88)] public LayerCq PhiPost;
            [FieldOffset(128)] public LayerCq PhiRes;
            [FieldOffset(168)] public int* PermA;
            [FieldOffset(176)] public int* PermB;
        }

        [StructLayout(LayoutKind.Explicit, Size = 128)]
        public struct LayerEngram
        {
            [FieldOffset(0)] public LayerCq Tables;
            [FieldOffset(40)] public LayerCq Key;
            [FieldOffset(80)] public LayerCq Value;
            [FieldOffset(120)] public float* Taps;
        }

        // Field-for-field mirror of EngineCore<NoClock>.LayerJob (private in
        // the core assembly). Bytes @8..11 are the NoClock clock plus padding
        // and are intentionally unmapped; JobSize pins the total.
        [StructLayout(LayoutKind.Explicit, Size = 1248)]
        public struct LayerJobMirror
        {
            [FieldOffset(0)] public LayerBarrier* Barrier;
            [FieldOffset(12)] public int Members;
            [FieldOffset(16)] public int Layer;
            [FieldOffset(20)] public int Lane;
            [FieldOffset(24)] public int Count;
            [FieldOffset(28)] public int Position;
            [FieldOffset(32)] public int Capacity;
            [FieldOffset(36)] public int HasEngram;
            [FieldOffset(40)] public LayerGeometry Geometry;
            [FieldOffset(160)] public LayerWeightsMirror Weights;
            [FieldOffset(536)] public LayerHyper Hyper;
            [FieldOffset(720)] public LayerEngram Engram;
            [FieldOffset(848)] public LayerActivations StreamAct;
            [FieldOffset(888)] public LayerActivations DimAct;
            [FieldOffset(928)] public float* Stream;
            [FieldOffset(936)] public float* Next;
            [FieldOffset(944)] public float* Normed;
            [FieldOffset(952)] public float* Input;
            [FieldOffset(960)] public float* Residual;
            [FieldOffset(968)] public float* Quantized;
            [FieldOffset(976)] public float* QRaw;
            [FieldOffset(984)] public float* KRaw;
            [FieldOffset(992)] public float* VRaw;
            [FieldOffset(1000)] public float* QConv;
            [FieldOffset(1008)] public float* KConv;
            [FieldOffset(1016)] public float* VConv;
            [FieldOffset(1024)] public float* GateOut;
            [FieldOffset(1032)] public float* AttnOut;
            [FieldOffset(1040)] public float* AttnProj;
            [FieldOffset(1048)] public float* EngramInput;
            [FieldOffset(1056)] public float* EngramKey;
            [FieldOffset(1064)] public float* EngramConv;
            [FieldOffset(1072)] public float* EngramValueBuffer;
            [FieldOffset(1080)] public float* QueryScale;
            [FieldOffset(1088)] public float* KeyScale;
            [FieldOffset(1096)] public float* ValueScale;
            [FieldOffset(1104)] public float* RopeCos;
            [FieldOffset(1112)] public float* RopeSin;
            [FieldOffset(1120)] public float* ResidualMix;
            [FieldOffset(1128)] public float* PostGate;
            [FieldOffset(1136)] public float* PreOut;
            [FieldOffset(1144)] public float* PostOut;
            [FieldOffset(1152)] public float* ResOut;
            [FieldOffset(1160)] public float* AttnPartials;
            [FieldOffset(1168)] public float* QkvBase;
            [FieldOffset(1176)] public float* EngramBase;
            [FieldOffset(1184)] public float* QkvLive;
            [FieldOffset(1192)] public float* EngramLive;
            [FieldOffset(1200)] public sbyte* QueryInt8;
            [FieldOffset(1208)] public sbyte* KeyCache;
            [FieldOffset(1216)] public sbyte* ValueCache;
            [FieldOffset(1224)] public int* EngramRows;
            [FieldOffset(1232)] public int* AttnArrivals;
            [FieldOffset(1240)] public BurstMlpWeights* Mlp;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct MatMulJobMirror
        {
            public LayerCq W;
            public float* Y;
            public int Ldy;
        }

        // ---- layout freeze: numeric half (attribute half is Engine.cs) ----

        // Probed Unsafe.SizeOf<LayerJob> for EngineCore<NoClock> (x64/arm64).
        public const int JobSize = 1248;

        // Managed-side check for N30's owner / core tests:
        // BurstLayer.AssertLayout(Unsafe.SizeOf<LayerJob>()).
        public static void AssertLayout(int engineJobSize)
        {
            if (engineJobSize != JobSize)
                throw new InvalidOperationException(
                    "N40: LayerJob size " + engineJobSize + " != BurstLayer.JobSize " +
                    JobSize + ". The engine struct moved; update BurstLayer mirrors.");
        }

        // ---- function pointer surface (managed side; N30 wires the table) ----

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LayerMemberFn(LayerJobMirror* job, int member);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int LaneFn();

        public static nint Function;

        static nint laneFp;

        public static bool IsInstalled => Function != 0;

        public static void Install()
        {
            if (Function != 0) return;
            FunctionPointer<LayerMemberFn> compiled =
                BurstCompiler.CompileFunctionPointer<LayerMemberFn>(Invoke);
            if (!compiled.IsCreated)
                throw new InvalidOperationException(
                    "N40: Burst refused to compile BurstLayer.Invoke. " +
                    "The kernel must live in package source (H6); " +
                    "a precompiled DLL entry cannot be compiled.");
            Function = (nint)compiled.Value;
            FunctionPointer<LaneFn> lane =
                BurstCompiler.CompileFunctionPointer<LaneFn>(Lane);
            if (!lane.IsCreated)
                throw new InvalidOperationException("N40: Burst refused to compile BurstLayer.Lane.");
            laneFp = (nint)lane.Value;
        }

        // 2 = x86 AVX2+FMA, 1 = arm Neon, 0 = scalar. Reads the COMPILED lane
        // through the pointer (a managed IsSupported read is always False).
        public static int CompiledLane()
        {
            if (laneFp == 0) return -1;
            return ((delegate* unmanaged[Cdecl]<int>)(void*)laneFp)();
        }

        public static string Describe() => "layer-fused=" + (Function != 0 ? "yes" : "no") + " lane=" + CompiledLane();

        // ---- the fused entry ----

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void Invoke(LayerJobMirror* job, int member)
        {
            int members = job->Members;
            HyperNorm(job, member, members);
            FusedBarrier(job->Barrier);
            HyperPhi(job, member, members);
            FusedBarrier(job->Barrier);
            HyperMix(job, member, members);
            if (job->HasEngram != 0)
            {
                EngramGather(job, member, members);
                FusedBarrier(job->Barrier);
                EngramMatMul(job, member, members);
                FusedBarrier(job->Barrier);
                EngramBlend(job, member, members);
            }
            AttentionPrepare(job, member, members);
            FusedBarrier(job->Barrier);
            AttentionQkvg(job, member, members);
            FusedBarrier(job->Barrier);
            TapsHistory(job, member, members);
            AttentionHeads(job, member, members);
            FusedBarrier(job->Barrier);
            AttentionCore(job, member, members);
            FusedBarrier(job->Barrier);
            AttentionGate(job, member, members);
            FusedBarrier(job->Barrier);
            AttentionOut(job, member, members);
            FusedBarrier(job->Barrier);
            PostNorm(job, member, members);
            Mlp(job, member, members);
            Combine(job, member, members);
        }

        // Compiled-lane reporter (same pattern as BurstSoftmax.Lane).
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static int Lane()
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) return 2;
            if (Arm.Neon.IsNeonSupported) return 1;
            return 0;
        }

        // Mirrors Pool.Barrier exactly (arrivals/phase handshake), with the
        // spin on Volatile phase backed by Common.Pause instead of
        // Spinner/SpinWait (managed thread helpers cannot run in Burst).
        static void FusedBarrier(LayerBarrier* barrier)
        {
            int participants = Volatile.Read(ref barrier->Participants);
            if (participants <= 1) return;
            long phase = Volatile.Read(ref barrier->Phase);
            if (Interlocked.Increment(ref barrier->Arrived) == participants)
            {
                Volatile.Write(ref barrier->Arrived, 0);
                Volatile.Write(ref barrier->Phase, phase + 1);
            }
            else
            {
                while (Volatile.Read(ref barrier->Phase) == phase)
                    Common.Pause();
            }
        }

        static void Shard(int tasks, int member, int members, out int first, out int count)
        {
            int per = (tasks + members - 1) / members;
            first = member * per < tasks ? member * per : tasks;
            int end = first + per < tasks ? first + per : tasks;
            count = end - first;
        }

        static float SigmoidInline(float x) => 1f / (1f + MathF.Exp(-x));

        static void CopyFloats(float* destination, float* source, int n)
        {
            for (int i = 0; i < n; i++) destination[i] = source[i];
        }

        // ---- matmul mini-scheduler (replicates MatMulBatch.Plan/RunTask) ----

        static void RunMatMulJobs(LayerActivations* act, int tokens, MatMulJobMirror* jobs, int jobCount, int member, int members)
        {
            int rows = 0;
            for (int j = 0; j < jobCount; j++) rows += jobs[j].W.Rows;
            int perTask = (rows + members * TasksPerThread - 1) / (members * TasksPerThread);
            if (perTask < MinimumRowsPerTask) perTask = MinimumRowsPerTask;
            int rowsPerTask = (perTask + RowBlock - 1) / RowBlock * RowBlock;
            int taskCount = 0;
            for (int j = 0; j < jobCount; j++) taskCount += (jobs[j].W.Rows + rowsPerTask - 1) / rowsPerTask;
            Shard(taskCount, member, members, out int first, out int count);
            for (int i = first; i < first + count; i++)
            {
                int index = i;
                for (int j = 0; j < jobCount; j++)
                {
                    int tasks = (jobs[j].W.Rows + rowsPerTask - 1) / rowsPerTask;
                    if (index >= tasks) { index -= tasks; continue; }
                    int start = index * rowsPerTask;
                    int end = jobs[j].W.Rows < start + rowsPerTask ? jobs[j].W.Rows : start + rowsPerTask;
                    MatMulShard(&jobs[j].W, act, tokens, jobs[j].Y, jobs[j].Ldy, start, end);
                    break;
                }
            }
        }

        static void MatMulShard(LayerCq* w, LayerActivations* a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
        {
            if (rowEnd <= rowStart) return;
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                BurstGemvTileX86.MatMulRows(
                    (GemvWeight*)w, (GemvActivations*)a,
                    tokens, y, ldy, rowStart, rowEnd);
                return;
            }
            if (Arm.Neon.IsNeonSupported)
            {
                BurstGemvNeon.Invoke(
                    (BurstGemvNeon.Cq*)w, (BurstGemvNeon.Activations*)a,
                    tokens, y, ldy, rowStart, rowEnd);
                return;
            }
            ScalarMatMulRows(w, a, tokens, y, ldy, rowStart, rowEnd);
        }

        static float HalfToSingleInline(ushort half)
        {
            int magnitude = half & 0x7FFF;
            if (magnitude == 0) return 0f;
            if (magnitude < 0x0400) return (half & 0x8000) != 0 ? -magnitude * 5.9604645e-8f : magnitude * 5.9604645e-8f;
            uint bits = ((uint)magnitude << 13) + 0x38000000u | (uint)(half & 0x8000) << 16;
            return *(float*)&bits;
        }

        // Exact port of Scalar.MatMulRows (sums in the same order; integer
        // dots are exact). Fallback lane where neither AVX2+FMA nor Neon
        // survives Burst compilation (WebGL-era targets).
        static void ScalarMatMulRows(LayerCq* w, LayerActivations* a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
        {
            int groups = w->Cols / 128;
            byte* table = w->Book + 64;
            float byteScale = *(float*)(w->Book + 96);
            int rowBytes = w->Cols * w->Bits / 8;
            for (int t = 0; t < tokens; t++)
            {
                sbyte* values = a->Values + (long)t * a->Cols;
                float* scales = a->Scales + t * a->Stride;
                float* offsets = a->Offsets + t * a->Stride;
                for (int n = rowStart; n < rowEnd; n++)
                {
                    byte* packed = w->Indices + (long)n * rowBytes;
                    ushort* norms = (ushort*)w->Norms + (long)n * groups;
                    float accumulator = 0f;
                    float bias = 0f;
                    for (int g = 0; g < groups; g++)
                    {
                        sbyte* x = values + g * 128;
                        int dot = 0;
                        if (w->Bits == 2)
                        {
                            byte* p = packed + g * 32;
                            for (int j = 0; j < 32; j++)
                            {
                                int b = p[j];
                                dot += table[b & 3] * x[j] + table[b >> 2 & 3] * x[32 + j] + table[b >> 4 & 3] * x[64 + j] + table[b >> 6 & 3] * x[96 + j];
                            }
                        }
                        else
                        {
                            byte* p = packed + g * 64;
                            for (int half = 0; half < 2; half++)
                                for (int j = 0; j < 32; j++)
                                {
                                    int b = p[half * 32 + j];
                                    dot += table[b & 15] * x[2 * half * 32 + j] + table[b >> 4] * x[(2 * half + 1) * 32 + j];
                                }
                        }
                        float scaled = HalfToSingleInline(norms[g]) * byteScale * scales[g];
                        accumulator += scaled * dot;
                        bias += scaled * offsets[g];
                    }
                    y[(long)t * ldy + n] = accumulator - bias;
                }
            }
        }

        // ---- stages (same order, same sharding, same calls as Engine.cs) ----

        static void HyperNorm(LayerJobMirror* job, int member, int members)
        {
            int streamWidth = Lanes * job->Geometry.Dim;
            Shard(job->Count, member, members, out int first, out int count);
            for (int t = first; t < first + count; t++)
            {
                float* x = job->Stream + (long)t * streamWidth;
                float* normed = job->Normed + (long)t * streamWidth;
                BurstRowOps.InvokeRmsFakeA8(x, streamWidth, normed);
                LayerActivations slice = Slice(job->StreamAct, t);
                BurstPrepare.InvokePrepare(normed, 1, streamWidth, 4, (BurstActivations*)&slice);
            }
        }

        static LayerActivations Slice(LayerActivations act, int token)
        {
            LayerActivations s = act;
            s.Values = act.Values + (long)token * act.Cols;
            s.Scales = act.Scales + token * act.Stride;
            s.Offsets = act.Offsets + token * act.Stride;
            return s;
        }

        // Mirrors Cq.Slice: per-layer row window into a shared codebook.
        static LayerCq SliceCq(LayerCq w, int first, int count)
        {
            int rowBytes = w.Cols * w.Bits / 8;
            int groups = w.Cols / 128;
            w.Indices = w.Indices + (long)first * rowBytes;
            w.Norms = w.Norms + (long)first * groups;
            w.Rows = count;
            return w;
        }

        static void HyperPhi(LayerJobMirror* job, int member, int members)
        {
            int l = job->Layer;
            MatMulJobMirror* jobs = stackalloc MatMulJobMirror[3];
            jobs[0].W = SliceCq(job->Hyper.PhiPre, l * Lanes, Lanes); jobs[0].Y = job->PreOut; jobs[0].Ldy = Lanes;
            jobs[1].W = SliceCq(job->Hyper.PhiPost, l * Lanes, Lanes); jobs[1].Y = job->PostOut; jobs[1].Ldy = Lanes;
            jobs[2].W = SliceCq(job->Hyper.PhiRes, l * Lanes * Lanes, Lanes * Lanes); jobs[2].Y = job->ResOut; jobs[2].Ldy = Lanes * Lanes;
            LayerActivations act = job->StreamAct;
            RunMatMulJobs(&act, job->Count, jobs, 3, member, members);
        }

        static void HyperMix(LayerJobMirror* job, int member, int members)
        {
            int dim = job->Geometry.Dim;
            int lane = job->Lane;
            int l = job->Layer;
            float* mixLogits = stackalloc float[Lanes * Lanes];
            float* hpre = stackalloc float[Lanes];
            Shard(job->Count, member, members, out int first, out int count);
            for (int t = first; t < first + count; t++)
            {
                float* s = job->Stream + (long)t * Lanes * dim;
                float* u = job->Input + (long)t * dim;
                for (int i = 0; i < Lanes; i++)
                {
                    hpre[i] = SigmoidInline(job->Hyper.APre[l] * job->PreOut[t * Lanes + i] + job->Hyper.BPre[l * Lanes + i] + (i == lane ? 4f : -4f));
                    job->PostGate[t * Lanes + i] = 2f * SigmoidInline(job->Hyper.APost[l] * job->PostOut[t * Lanes + i] + job->Hyper.BPost[l * Lanes + i] + (i == lane ? 0f : -4f));
                }
                BurstRowOps.InvokeMix4(s, dim, hpre[0], hpre[1], hpre[2], hpre[3], u);
                for (int i = 0; i < Lanes * Lanes; i++) mixLogits[i] = job->Hyper.ARes[l] * job->ResOut[t * Lanes * Lanes + i] + job->Hyper.BRes[l * Lanes * Lanes + i];
                BurstRowOps.InvokeSinkhorn4x4(mixLogits, job->ResidualMix + (long)t * Lanes * Lanes);
                CopyFloats(job->Residual + (long)t * dim, u, dim);
            }
        }

        static void EngramGather(LayerJobMirror* job, int member, int members)
        {
            int dim = job->Geometry.Dim;
            int tables = job->Geometry.EngramTables;
            int sub = job->Geometry.EngramSub;
            Shard(job->Count, member, members, out int first, out int count);
            for (int t = first; t < first + count; t++)
            {
                float* e = job->EngramInput + (long)t * dim;
                for (int table = 0; table < tables; table++)
                {
                    int row = job->EngramRows[t * tables + table];
                    float* dst = e + table * sub;
                    if (row < 0) { for (int i = 0; i < sub; i++) dst[i] = 0f; }
                    else
                    {
                        LayerCq w = job->Engram.Tables;
                        BurstMisc.GatherRow((BurstMisc.BurstCq*)&w, row, dst);
                    }
                }
                BurstRowOps.InvokeFakeA8(e, dim);
                LayerActivations slice = Slice(job->DimAct, t);
                BurstPrepare.InvokePrepare(e, 1, dim, 2, (BurstActivations*)&slice);
            }
            if (member == 0) CopyFloats(job->EngramValueBuffer, job->EngramBase, EngramHistory * dim);
        }

        static void EngramMatMul(LayerJobMirror* job, int member, int members)
        {
            MatMulJobMirror* jobs = stackalloc MatMulJobMirror[2];
            jobs[0].W = job->Engram.Key; jobs[0].Y = job->EngramKey; jobs[0].Ldy = job->Geometry.Dim;
            jobs[1].W = job->Engram.Value; jobs[1].Y = job->EngramValueBuffer + EngramHistory * job->Geometry.Dim; jobs[1].Ldy = job->Geometry.Dim;
            LayerActivations act = job->DimAct;
            RunMatMulJobs(&act, job->Count, jobs, 2, member, members);
        }

        static void EngramBlend(LayerJobMirror* job, int member, int members)
        {
            int dim = job->Geometry.Dim;
            float* unitA = stackalloc float[dim];
            float* unitB = stackalloc float[dim];
            float inverseRootDim = 1f / MathF.Sqrt(dim);
            Shard(job->Count, member, members, out int first, out int count);
            for (int t = first; t < first + count; t++)
            {
                float* o = job->EngramConv + (long)t * dim;
                for (int i = 0; i < dim; i++) o[i] = 0f;
                for (int j = 0; j < EngramTaps; j++)
                    BurstRowOps.InvokeMultiplyAccumulate(o, job->Engram.Taps + j * dim, job->EngramValueBuffer + (long)(EngramHistory + t - j * job->Geometry.EngramDilation) * dim, dim);
            }
            if (member == 0) CopyFloats(job->EngramLive, job->EngramValueBuffer + (long)job->Count * dim, EngramHistory * dim);
            for (int t = first; t < first + count; t++)
            {
                float* x = job->Input + (long)t * dim;
                BurstRowOps.InvokeRmsUnit(x, dim, unitA);
                BurstRowOps.InvokeRmsUnit(job->EngramKey + (long)t * dim, dim, unitB);
                float alpha = SigmoidInline(BurstRowOps.InvokeDot(unitA, unitB, dim) * inverseRootDim);
                BurstRowOps.InvokeAddScaled(job->Residual + (long)t * dim, job->EngramConv + (long)t * dim, alpha, dim);
            }
        }

        static void AttentionPrepare(LayerJobMirror* job, int member, int members)
        {
            int dim = job->Geometry.Dim;
            Shard(job->Count, member, members, out int first, out int count);
            for (int t = first; t < first + count; t++)
            {
                float* q = job->Quantized + (long)t * dim;
                BurstRowOps.InvokeCenteredRmsFakeA8(job->Residual + (long)t * dim, job->Weights.NormIn, dim, q);
                LayerActivations slice = Slice(job->DimAct, t);
                BurstPrepare.InvokePrepare(q, 1, dim, 2, (BurstActivations*)&slice);
            }
        }

        static void AttentionQkvg(LayerJobMirror* job, int member, int members)
        {
            int qWidth = job->Geometry.Heads * job->Geometry.QkDim;
            int kWidth = job->Geometry.KvHeads * job->Geometry.QkDim;
            int vWidth = job->Geometry.KvHeads * job->Geometry.VDim;
            int outWidth = job->Geometry.Heads * job->Geometry.VDim;
            MatMulJobMirror* jobs = stackalloc MatMulJobMirror[4];
            jobs[0].W = job->Weights.Q; jobs[0].Y = job->QRaw + (ConvTaps - 1) * qWidth; jobs[0].Ldy = qWidth;
            jobs[1].W = job->Weights.K; jobs[1].Y = job->KRaw + (ConvTaps - 1) * kWidth; jobs[1].Ldy = kWidth;
            jobs[2].W = job->Weights.V; jobs[2].Y = job->VRaw + (ConvTaps - 1) * vWidth; jobs[2].Ldy = vWidth;
            jobs[3].W = job->Weights.Gate; jobs[3].Y = job->GateOut; jobs[3].Ldy = outWidth;
            LayerActivations act = job->DimAct;
            RunMatMulJobs(&act, job->Count, jobs, 4, member, members);
        }

        static void TapsHistory(LayerJobMirror* job, int member, int members)
        {
            int qWidth = job->Geometry.Heads * job->Geometry.QkDim;
            int kWidth = job->Geometry.KvHeads * job->Geometry.QkDim;
            int vWidth = job->Geometry.KvHeads * job->Geometry.VDim;
            float* qHistory = job->QkvBase;
            float* kHistory = qHistory + (ConvTaps - 1) * qWidth;
            float* vHistory = kHistory + (ConvTaps - 1) * kWidth;
            if (member == 0)
            {
                CopyFloats(job->QRaw, qHistory, (ConvTaps - 1) * qWidth);
                CopyFloats(job->KRaw, kHistory, (ConvTaps - 1) * kWidth);
                CopyFloats(job->VRaw, vHistory, (ConvTaps - 1) * vWidth);
            }
            FusedBarrier(job->Barrier);
            Shard(job->Count, member, members, out int first, out int count);
            for (int t = first; t < first + count; t++)
            {
                BurstRowOps.InvokeTaps3(job->Weights.QTaps, qWidth, job->QRaw + (long)(t + ConvTaps - 1) * qWidth, job->QConv + (long)t * qWidth);
                BurstRowOps.InvokeTaps3(job->Weights.KTaps, kWidth, job->KRaw + (long)(t + ConvTaps - 1) * kWidth, job->KConv + (long)t * kWidth);
                BurstRowOps.InvokeTaps3(job->Weights.VTaps, vWidth, job->VRaw + (long)(t + ConvTaps - 1) * vWidth, job->VConv + (long)t * vWidth);
            }
            FusedBarrier(job->Barrier);
            if (member == 0)
            {
                float* qLive = job->QkvLive;
                float* kLive = qLive + (ConvTaps - 1) * qWidth;
                float* vLive = kLive + (ConvTaps - 1) * kWidth;
                CopyFloats(qLive, job->QRaw + (long)job->Count * qWidth, (ConvTaps - 1) * qWidth);
                CopyFloats(kLive, job->KRaw + (long)job->Count * kWidth, (ConvTaps - 1) * kWidth);
                CopyFloats(vLive, job->VRaw + (long)job->Count * vWidth, (ConvTaps - 1) * vWidth);
            }
        }

        static void AttentionHeads(LayerJobMirror* job, int member, int members)
        {
            int heads = job->Geometry.Heads;
            int kvHeads = job->Geometry.KvHeads;
            int qkDim = job->Geometry.QkDim;
            int vDim = job->Geometry.VDim;
            int half = qkDim / 2;
            sbyte* layerKeys = job->KeyCache + (long)job->Layer * job->Capacity * job->Geometry.KeyWidth;
            sbyte* layerValues = job->ValueCache + (long)job->Layer * job->Capacity * job->Geometry.ValueWidth;
            float* layerKeyScales = job->KeyScale + (long)job->Layer * job->Capacity * kvHeads;
            float* layerValueScales = job->ValueScale + (long)job->Layer * job->Capacity * kvHeads;
            Shard(job->Count * (heads + kvHeads), member, members, out int first, out int count);
            for (int u = first; u < first + count; u++)
            {
                int t = u / (heads + kvHeads);
                int h = u % (heads + kvHeads);
                int p = job->Position + t;
                float* cos = job->RopeCos + (long)p * half;
                float* sin = job->RopeSin + (long)p * half;
                if (h < heads)
                {
                    float* q = job->QConv + ((long)t * heads + h) * qkDim;
                    BurstRowOps.InvokeZeroCenteredRms(q, job->Weights.QNorm, qkDim, q);
                    BurstRowOps.InvokeRope(q, 1, qkDim, cos, sin);
                    job->QueryScale[(long)t * heads + h] = BurstRowOps.InvokeQuantizeInt8(q, qkDim, job->QueryInt8 + ((long)t * heads + h) * qkDim);
                }
                else
                {
                    int k = h - heads;
                    float* kv = job->KConv + ((long)t * kvHeads + k) * qkDim;
                    BurstRowOps.InvokeZeroCenteredRms(kv, job->Weights.KNorm, qkDim, kv);
                    BurstRowOps.InvokeRope(kv, 1, qkDim, cos, sin);
                    layerKeyScales[(long)p * kvHeads + k] = BurstRowOps.InvokeQuantizeInt8(kv, qkDim, layerKeys + ((long)p * kvHeads + k) * qkDim);
                    float* v = job->VConv + ((long)t * kvHeads + k) * vDim;
                    layerValueScales[(long)p * kvHeads + k] = BurstRowOps.InvokeQuantizeInt8(v, vDim, layerValues + ((long)p * kvHeads + k) * vDim);
                }
            }
        }

        static void AttentionCore(LayerJobMirror* job, int member, int members)
        {
            int heads = job->Geometry.Heads;
            int kvHeads = job->Geometry.KvHeads;
            int group = heads / kvHeads;
            int tilesPerGroup = (group + MaxTileHeads - 1) / MaxTileHeads;
            int tileHeads = (group + tilesPerGroup - 1) / tilesPerGroup;
            int planGroups = job->Count * kvHeads;
            // CanSplitPositions in Burst terms: the x86 AVX2+FMA lane splits
            // positions across chunks (net10 Simd.UseAvx equivalent).
            bool splittable = X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported && tilesPerGroup == 1 && group > 1;
            int chunks = splittable ? (members + planGroups - 1) / planGroups : 1;
            if (chunks < 1) chunks = 1;
            if (chunks > MaxChunks) chunks = MaxChunks;
            int teamTasks = planGroups * (chunks == 1 ? tilesPerGroup : chunks);
            if (member == 0) for (int g = 0; g < planGroups; g++) job->AttnArrivals[g] = 0;
            FusedBarrier(job->Barrier);
            Shard(teamTasks, member, members, out int first, out int count);
            for (int i = first; i < first + count; i++)
            {
                if (chunks == 1) ExecuteAttentionTile(job, i, tilesPerGroup, tileHeads, kvHeads);
                else ExecuteAttentionChunk(job, i, chunks, kvHeads, group);
            }
        }

        static void AttentionSpan(LayerJobMirror* job, int token, out int first, out int count)
        {
            int position = job->Position + token;
            bool global = ((job->Geometry.GlobalMask >> job->Layer) & 1) != 0;
            int window = global ? 0 : job->Geometry.SlidingWindow;
            first = window == 0 ? 0 : (position - window + 1 > 0 ? position - window + 1 : 0);
            count = position - first + 1; // Frames is 0 on this path
        }

        static void FillTile(LayerJobMirror* job, BurstAttentionTile* work, int token, int kv, int firstHead, int headCount, int first, int count, float* scores)
        {
            int heads = job->Geometry.Heads;
            int kvHeads = job->Geometry.KvHeads;
            int qkDim = job->Geometry.QkDim;
            int vDim = job->Geometry.VDim;
            sbyte* layerKeys = job->KeyCache + (long)job->Layer * job->Capacity * job->Geometry.KeyWidth;
            sbyte* layerValues = job->ValueCache + (long)job->Layer * job->Capacity * job->Geometry.ValueWidth;
            float* layerKeyScales = job->KeyScale + (long)job->Layer * job->Capacity * kvHeads;
            float* layerValueScales = job->ValueScale + (long)job->Layer * job->Capacity * kvHeads;
            work->Queries = job->QueryInt8 + (long)token * heads * qkDim + firstHead * qkDim;
            work->QueryScales = job->QueryScale + token * heads + firstHead;
            work->Heads = headCount;
            work->Keys = layerKeys + kv * qkDim;
            work->KeyScales = layerKeyScales + kv;
            work->Values = layerValues + kv * vDim;
            work->ValueScales = layerValueScales + kv;
            work->KeyStride = kvHeads * qkDim;
            work->ValueStride = kvHeads * vDim;
            work->ScaleStride = kvHeads;
            work->First = first;
            work->Count = count;
            work->Scores = scores;
            work->Output = null;
            work->Probabilities = null;
            work->ProbabilityStride = 0;
            work->Peaks = null;
            work->Sums = null;
        }

        static void ExecuteAttentionTile(LayerJobMirror* job, int task, int tilesPerGroup, int tileHeads, int kvHeads)
        {
            int heads = job->Geometry.Heads;
            int vDim = job->Geometry.VDim;
            int group = heads / kvHeads;
            int token = task / (kvHeads * tilesPerGroup);
            int kv = task / tilesPerGroup % kvHeads;
            int firstHead = kv * group + task % tilesPerGroup * tileHeads;
            AttentionSpan(job, token, out int first, out int count);
            float* scores = stackalloc float[(count + 4) * 8];
            BurstAttentionTile work = new BurstAttentionTile();
            int headCount = tileHeads < (kv + 1) * group - firstHead ? tileHeads : (kv + 1) * group - firstHead;
            FillTile(job, &work, token, kv, firstHead, headCount, first, count, scores);
            work.Output = job->AttnOut + (long)token * heads * vDim + firstHead * vDim;
            BurstAttention.Tile(&work);
        }

        static void ExecuteAttentionChunk(LayerJobMirror* job, int task, int chunks, int kvHeads, int group)
        {
            int heads = job->Geometry.Heads;
            int vDim = job->Geometry.VDim;
            int chunk = task % chunks;
            int groupIndex = task / chunks;
            int token = groupIndex / kvHeads;
            int kv = groupIndex % kvHeads;
            AttentionSpan(job, token, out int first, out int count);
            int needed = count / MinChunkPositions;
            if (needed < 1) needed = 1;
            if (needed > chunks) needed = chunks;
            int length = ((count + needed - 1) / needed + 3) & ~3;
            needed = (count + length - 1) / length;
            if (chunk >= needed) return;
            int begin = chunk * length;
            int rest = count - begin;
            int take = rest < length ? rest : length;
            float* slot = job->AttnPartials + (long)(groupIndex * chunks + chunk) * PartialFloats;
            float* scores = stackalloc float[(take + 4) * 8];
            BurstAttentionTile work = new BurstAttentionTile();
            FillTile(job, &work, token, kv, kv * group, group, first + begin, take, scores);
            work.Output = slot;
            work.Peaks = slot + PeaksOffset;
            work.Sums = slot + SumsOffset;
            BurstAttention.Tile(&work);
            if (Interlocked.Increment(ref job->AttnArrivals[groupIndex]) == needed)
                BurstAttention.Merge(job->AttnPartials + (long)groupIndex * chunks * PartialFloats, needed, group, job->AttnOut + (long)token * heads * vDim + kv * group * vDim);
        }

        static void AttentionGate(LayerJobMirror* job, int member, int members)
        {
            int outWidth = job->Geometry.Heads * job->Geometry.VDim;
            Shard(job->Count, member, members, out int first, out int count);
            for (int t = first; t < first + count; t++)
            {
                float* o = job->AttnOut + (long)t * outWidth;
                BurstRowOps.InvokeGateFakeA8(o, job->GateOut + (long)t * outWidth, outWidth);
                LayerActivations slice = Slice(job->DimAct, t);
                BurstPrepare.InvokePrepare(o, 1, outWidth, 2, (BurstActivations*)&slice);
            }
        }

        static void AttentionOut(LayerJobMirror* job, int member, int members)
        {
            MatMulJobMirror* jobs = stackalloc MatMulJobMirror[1];
            jobs[0].W = job->Weights.Out; jobs[0].Y = job->AttnProj; jobs[0].Ldy = job->Geometry.Dim;
            LayerActivations act = job->DimAct;
            RunMatMulJobs(&act, job->Count, jobs, 1, member, members);
        }

        static void PostNorm(LayerJobMirror* job, int member, int members)
        {
            int dim = job->Geometry.Dim;
            Shard(job->Count, member, members, out int first, out int count);
            for (int t = first; t < first + count; t++)
                BurstRowOps.InvokePostNormAdd(job->Residual + (long)t * dim, job->AttnProj + (long)t * dim, job->Weights.PostNorm, job->Weights.AttnGate, dim);
        }

        static void Mlp(LayerJobMirror* job, int member, int members)
        {
            int dim = job->Geometry.Dim;
            int hadaN = job->Geometry.HadaN;
            int bits = 0;
            for (int v = hadaN - 1; v != 0; v >>= 1) bits++;
            int rows = 1 << (bits / 2);
            int cols = hadaN / rows;
            BurstMlpBatch b = new BurstMlpBatch();
            b.Weights = job->Mlp;
            b.PreNorm = job->Weights.PreHada;
            b.PermA = job->Hyper.PermA;
            b.PermB = job->Hyper.PermB;
            b.Residual = job->Residual;
            b.Dim = dim;
            b.Rows = rows;
            b.Cols = cols;
            b.Scale = 1f;
            int width = rows * cols;
            Shard(job->Count, member, members, out int first, out int count);
            for (int t = first; t < first + count; t++)
            {
                float* scratch = stackalloc float[8 + dim + 2 * width + 8];
                BurstMlp.Invoke(&b, job->Residual + (long)t * dim, scratch);
            }
        }

        static void Combine(LayerJobMirror* job, int member, int members)
        {
            int dim = job->Geometry.Dim;
            Shard(job->Count, member, members, out int first, out int count);
            for (int t = first; t < first + count; t++)
            {
                float* s = job->Stream + (long)t * Lanes * dim;
                float* n = job->Next + (long)t * Lanes * dim;
                float* x = job->Residual + (long)t * dim;
                float* u = job->Input + (long)t * dim;
                float* mix = job->ResidualMix + (long)t * Lanes * Lanes;
                for (int i = 0; i < Lanes; i++)
                    BurstRowOps.InvokeCombineDiff4(s, dim, mix[i * 4], mix[i * 4 + 1], mix[i * 4 + 2], mix[i * 4 + 3], x, u, job->PostGate[t * Lanes + i], n + (long)i * dim);
            }
        }
    }
}
