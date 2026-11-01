using System;

namespace Needle;

internal sealed unsafe class Vocabulary : IDisposable
{
    internal readonly Tokenizer Tables;
    internal readonly TokenTrie Trie;
    internal readonly uint* ReasoningMask;
    internal readonly int MaskWords;

    internal Vocabulary(Model model)
    {
        Tables = new Tokenizer(model, 1);
        Trie = new TokenTrie(Tables, model.VocabSize);
        MaskWords = Trie.Words;
        ReasoningMask = (uint*)Mem.AllocZeroed((nuint)MaskWords * 4);
        for (var id = 0; id < Math.Min(model.VocabSize, Tables.PieceCount); id++)
            if (Tables.IsConstrainable(id)) ReasoningMask[id / 32] |= 1u << (id % 32);
        ReasoningMask[7 / 32] |= 1u << (7 % 32);
    }

    public void Dispose()
    {
        Mem.Free(ReasoningMask);
        Trie.Dispose();
        Tables.Dispose();
    }
}
