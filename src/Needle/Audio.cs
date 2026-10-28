using System;

namespace Needle;

internal readonly unsafe struct MlpWeights
{
    public readonly float* D1, D2, B2, D3, D4;
    public readonly float* W1A, W1B, W2A, W2B, W3A, W3B;
    public readonly float* CondV, CondU;

    public MlpWeights(ref Reader r)
    {
        D1 = r.Dense();
        D2 = r.Dense();
        B2 = r.Dense();
        D3 = r.Dense();
        D4 = r.Dense();
        W1A = r.Dense();
        W1B = r.Dense();
        W2A = r.Dense();
        W2B = r.Dense();
        W3A = r.Dense();
        W3B = r.Dense();
        CondV = r.Dense();
        CondU = r.Dense();
    }
}

internal readonly unsafe struct ProbeHead
{
    public const int Embedding = 1, Confidence = 2, Router = 3;

    public readonly int Code, Probes, Queries, Outputs, Rows, Dim;
    public readonly float* ProbeVectors, Gain, QueryVectors, RowBias, Projection, Bias;

    public ProbeHead(ref Reader r, int code, int layers, int dim)
    {
        Code = code;
        Dim = dim;
        var probes = r.Quantized();
        Rows = layers + 1;
        Probes = probes.Rows / Rows;
        ProbeVectors = r.Rows(probes);
        Gain = r.Dense();
        var query = r.Quantized();
        Queries = query.Rows;
        QueryVectors = r.Rows(query);
        RowBias = r.Dense();
        var projection = r.Quantized();
        Outputs = projection.Rows;
        Projection = r.Rows(projection);
        Bias = r.Dense();
        if (code == Router) r.Dense();
    }
}

internal readonly unsafe struct CrossWeights
{
    public readonly float* Norm, QNorm, KNorm, PostNorm;
    public readonly Cq Q, K, V, Gate, Out;
    public readonly float Mix;

    public CrossWeights(ref Reader r)
    {
        Norm = r.Dense();
        Q = r.Quantized();
        K = r.Quantized();
        V = r.Quantized();
        Gate = r.Quantized();
        Out = r.Quantized();
        QNorm = r.Dense();
        KNorm = r.Dense();
        PostNorm = r.Dense();
        Mix = 1f / (1f + MathF.Exp(-*r.Dense()));
    }
}

internal readonly unsafe struct EncoderLayer
{
    public readonly float* PreMlpNorm0, AttnNorm, PostAttnNorm, ConvNorm, ConvOutNorm, PreMlpNorm;
    public readonly Cq Q, K, V, Gate, Out, Pw1, Pw2;
    public readonly float* QNorm, KNorm, Depthwise;
    public readonly float AttnGate;
    public readonly MlpWeights Mlp0, Mlp;

    public EncoderLayer(ref Reader r)
    {
        PreMlpNorm0 = r.Dense();
        AttnNorm = r.Dense();
        PostAttnNorm = r.Dense();
        ConvNorm = r.Dense();
        ConvOutNorm = r.Dense();
        PreMlpNorm = r.Dense();
        Q = r.Quantized();
        K = r.Quantized();
        V = r.Quantized();
        Gate = r.Quantized();
        Out = r.Quantized();
        QNorm = r.Dense();
        KNorm = r.Dense();
        AttnGate = 1f / (1f + MathF.Exp(-*r.Dense()));
        Pw1 = r.Quantized();
        Pw2 = r.Quantized();
        Depthwise = r.Dense();
        Mlp0 = new MlpWeights(ref r);
        Mlp = new MlpWeights(ref r);
    }
}

internal readonly unsafe struct AudioWeights
{
    public readonly int EncoderLayers, EncoderLanes, ConvKernel, StemWidth, StemStages, Mels, SampleRate, Fft, Window, Hop;
    public readonly CrossWeights* Cross;
    public readonly float PositionMix;
    public readonly EncoderLayer* Encoder;
    public readonly HyperConnections EncoderHyper;
    public readonly float* EncoderFinalNorm;
    public readonly float* StemFirst, StemDepthwise1, StemDepthwise2;
    public readonly Cq StemPointwise1, StemPointwise2, StemOut;
    public readonly float* Mel;

    public AudioWeights(ref Reader r, int decoderLayers, int hadaLength)
    {
        var manifest = r.Dense();
        EncoderLayers = (int)manifest[3];
        EncoderLanes = (int)manifest[4];
        ConvKernel = (int)manifest[5];
        StemWidth = (int)manifest[6];
        StemStages = (int)manifest[7];
        Mels = (int)manifest[8];
        SampleRate = (int)manifest[9];
        Fft = (int)manifest[10];
        Window = (int)manifest[11];
        Hop = (int)manifest[12];
        if (StemStages != 3) throw new NotSupportedException("three-stage stem expected");
        Cross = r.Take<CrossWeights>((nuint)decoderLayers);
        for (var l = 0; l < decoderLayers; l++) Cross[l] = new CrossWeights(ref r);
        PositionMix = 1f / (1f + MathF.Exp(-*r.Dense()));
        Encoder = r.Take<EncoderLayer>((nuint)EncoderLayers);
        for (var l = 0; l < EncoderLayers; l++) Encoder[l] = new EncoderLayer(ref r);
        EncoderHyper = new HyperConnections(ref r, 0);
        EncoderFinalNorm = r.Dense();
        StemFirst = r.Dense();
        StemDepthwise1 = r.Dense();
        StemPointwise1 = r.Quantized();
        StemDepthwise2 = r.Dense();
        StemPointwise2 = r.Quantized();
        StemOut = r.Quantized();
        Mel = r.Dense();
    }
}
