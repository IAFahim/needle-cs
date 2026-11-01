using System;

namespace Needle;

public readonly struct ModelInfo
{
    public const uint SupportedMagic = 0x05E12A84;
    public const int HeaderBytes = 196;
    public const string RefreshGuidance = "If the upstream model changed, see CONTRIBUTING.md, section \"Refreshing the upstream model\".";

    public readonly uint Magic;
    public readonly long FileBytes;
    public readonly int TensorCount;
    public readonly int CodebookEntries;
    public readonly int VocabSize;
    public readonly int HiddenSize;
    public readonly int LayerCount;
    public readonly int HeadCount;
    public readonly int KeyValueHeadCount;
    public readonly int QueryKeyHeadDim;
    public readonly int ValueHeadDim;
    public readonly int MaxPositions;
    public readonly int SlidingWindow;
    public readonly int EngramSiteCount;
    public readonly float RopeTheta;
    public readonly long TokenizerBytes;
    public readonly bool HasAudioHead;
    public readonly bool HasConfidenceHead;
    public readonly bool HasEmbeddingHead;

    internal ModelInfo(in Geometry geometry, uint magic, long fileBytes, int tensorCount, int codebookEntries, long tokenizerBytes, bool hasAudioHead, bool hasConfidenceHead, bool hasEmbeddingHead)
    {
        Magic = magic;
        FileBytes = fileBytes;
        TensorCount = tensorCount;
        CodebookEntries = codebookEntries;
        VocabSize = geometry.Vocab;
        HiddenSize = geometry.Dim;
        LayerCount = geometry.Layers;
        HeadCount = geometry.Heads;
        KeyValueHeadCount = geometry.KvHeads;
        QueryKeyHeadDim = geometry.QkDim;
        ValueHeadDim = geometry.VDim;
        MaxPositions = geometry.MaxSeq;
        SlidingWindow = geometry.SlidingWindow;
        EngramSiteCount = geometry.SiteCount;
        RopeTheta = geometry.RopeTheta;
        TokenizerBytes = tokenizerBytes;
        HasAudioHead = hasAudioHead;
        HasConfidenceHead = hasConfidenceHead;
        HasEmbeddingHead = hasEmbeddingHead;
    }

    public bool HasTokenizer => TokenizerBytes > 0;

    public override string ToString() =>
        FormattableString.Invariant(
            $"magic 0x{Magic:X8}, {FileBytes} bytes, {TensorCount} tensors, vocab {VocabSize}, hidden {HiddenSize}, layers {LayerCount}, heads {HeadCount}/{KeyValueHeadCount}, qk {QueryKeyHeadDim}, v {ValueHeadDim}, positions {MaxPositions}, window {SlidingWindow}, engram sites {EngramSiteCount}, rope theta {RopeTheta}, tokenizer {TokenizerBytes} bytes, audio head {HasAudioHead}, confidence head {HasConfidenceHead}, embedding head {HasEmbeddingHead}");
}
