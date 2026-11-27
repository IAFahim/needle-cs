using System;

namespace Needle;

internal sealed class RepairText
{
    public const int MaxNumbers = 128;

    public readonly byte[] Source;
    public readonly byte[] Norm;
    public readonly int[] Origin;
    public readonly int[] WordStart;
    public readonly int[] WordEnd;
    public readonly int[] WordClause;
    public readonly double[] Numbers;
    public readonly int[] NumberWord;
    public int SourceLength;
    public int Length;
    public int Words;
    public int Clauses;
    public int NumberCount;

    public RepairText(int capacity)
    {
        capacity = Math.Max(16, capacity);
        Source = new byte[capacity];
        Norm = new byte[capacity];
        Origin = new int[capacity];
        WordStart = new int[capacity / 2 + 1];
        WordEnd = new int[capacity / 2 + 1];
        WordClause = new int[capacity / 2 + 1];
        Numbers = new double[MaxNumbers];
        NumberWord = new int[MaxNumbers];
    }

    public int Capacity => Source.Length;

    /// <summary>Bytes of Source that came from Load (the system text in a conversation history); the
    /// turns appended after it follow.</summary>
    public int Preamble;

    public void Clear()
    {
        SourceLength = 0;
        Preamble = 0;
        Length = 0;
        Words = 0;
        Clauses = 0;
        NumberCount = 0;
    }

    public void Load(ReadOnlySpan<byte> source)
    {
        SourceLength = 0;
        Preamble = 0;
        Append(source);
        Preamble = SourceLength;
    }

    public void Append(ReadOnlySpan<byte> source)
    {
        var limit = Source.Length - 1;
        if (source.Length > limit) source = source[..limit];
        var need = SourceLength + 1 + source.Length - limit;
        if (need > 0)
        {
            // Full: drop the oldest text, on a line boundary, so the newest turn is always kept.
            if (need >= SourceLength) { SourceLength = 0; Preamble = 0; }
            else
            {
                var cut = Array.IndexOf(Source, (byte)'\n', need, SourceLength - need);
                cut = cut < 0 ? need : cut + 1;
                Buffer.BlockCopy(Source, cut, Source, 0, SourceLength - cut);
                SourceLength -= cut;
                Preamble = Math.Max(0, Preamble - cut);
            }
        }
        if (SourceLength > 0) Source[SourceLength++] = (byte)'\n';
        source.CopyTo(Source.AsSpan(SourceLength));
        SourceLength += source.Length;
        Analyze();
    }

    void Analyze()
    {
        Length = 0;
        Words = 0;
        Clauses = SourceLength > 0 ? 1 : 0;
        NumberCount = 0;
        negationClause = -1;
        var src = new ReadOnlySpan<byte>(Source, 0, SourceLength);
        var breakPending = false;
        var inWord = false;
        for (var i = 0; i < src.Length; i++)
        {
            var c = src[i];
            if (c == (byte)'\\' && i + 1 < src.Length)
            {
                i++;
                if (src[i] == (byte)'u') i = Math.Min(src.Length - 1, i + 4);
                EndWord(ref inWord);
                continue;
            }
            if (c == 0xE2 && i + 2 < src.Length && src[i + 1] == 0x80)
            {
                if (src[i + 2] == 0x99 && inWord && i + 3 < src.Length && IsLetter(src[i + 3])) { i += 2; continue; }
                i += 2;
                EndWord(ref inWord);
                continue;
            }
            byte mapped;
            if (c >= (byte)'A' && c <= (byte)'Z') mapped = (byte)(c | 0x20);
            else if ((c >= (byte)'a' && c <= (byte)'z') || (c >= (byte)'0' && c <= (byte)'9') || c >= 0x80) mapped = c;
            else if (c == (byte)'\'' && inWord && i + 1 < src.Length && IsLetter(src[i + 1])) continue;
            else if (c == (byte)',' && inWord && IsDigit(src[i - 1]) && ThousandsGroup(src, i + 1)) continue;
            else if (c == (byte)'.' && inWord && IsDigit(src[i - 1]) && i + 1 < src.Length && IsDigit(src[i + 1])) mapped = c;
            else
            {
                // "Dr. Patel" is one name, not two clauses: a period after a title abbreviation does not end a clause.
                if ((c is (byte)',' or (byte)';' or (byte)'.' or (byte)'!' or (byte)'?' or (byte)':' or (byte)'\n' or (byte)'(' or (byte)')') &&
                    !(c == (byte)'.' && inWord && IsTitle(new ReadOnlySpan<byte>(Norm, WordStart[Words], Length - WordStart[Words]))))
                    breakPending = true;
                EndWord(ref inWord);
                continue;
            }
            if (!inWord)
            {
                if (Length > 0) { Norm[Length] = (byte)' '; Origin[Length] = i; Length++; }
                if (breakPending && Words > 0) Clauses++;
                breakPending = false;
                WordStart[Words] = Length;
                WordClause[Words] = Clauses - 1;
                inWord = true;
            }
            Norm[Length] = mapped;
            Origin[Length] = i;
            Length++;
        }
        EndWord(ref inWord);
        CollectNumbers();
    }

    void EndWord(ref bool inWord)
    {
        if (!inWord) return;
        WordEnd[Words] = Length;
        var word = Word(Words);
        if (IsNegation(word)) negationClause = Clauses - 1;
        if (IsConjunction(word) && Words > 0 && !((word.SequenceEqual("or"u8) || word.SequenceEqual("nor"u8)) && negationClause == WordClause[Words]))
        {
            WordClause[Words] = Clauses;
            Clauses++;
        }
        Words++;
        inWord = false;
    }

    int negationClause = -1;

    // "don't turn on the lights or the fan" negates both, so "or" after a negation does not start a new clause.
    static bool IsNegation(ReadOnlySpan<byte> word) =>
        word.SequenceEqual("dont"u8) || word.SequenceEqual("not"u8) || word.SequenceEqual("never"u8) || word.SequenceEqual("neither"u8) ||
        word.SequenceEqual("wont"u8) || word.SequenceEqual("shouldnt"u8) || word.SequenceEqual("cant"u8);

    static bool IsTitle(ReadOnlySpan<byte> word) =>
        word.SequenceEqual("dr"u8) || word.SequenceEqual("mr"u8) || word.SequenceEqual("mrs"u8) || word.SequenceEqual("ms"u8) ||
        word.SequenceEqual("prof"u8) || word.SequenceEqual("st"u8) || word.SequenceEqual("mt"u8) || word.SequenceEqual("jr"u8) || word.SequenceEqual("sr"u8);

    // A word of a value counts as present when the text holds it, or holds it split in two or three
    // adjacent words ("6 pm" grounds "6pm").
    public bool HasWordOrJoined(ReadOnlySpan<byte> word)
    {
        if (HasWord(word)) return true;
        for (var w = 0; w < Words; w++)
        {
            var at = 0;
            for (var k = w; k < Words && k < w + 3; k++)
            {
                var piece = Word(k);
                if (at + piece.Length > word.Length || !word.Slice(at, piece.Length).SequenceEqual(piece)) break;
                at += piece.Length;
                if (at == word.Length) return k > w;
            }
        }
        return false;
    }

    static bool IsConjunction(ReadOnlySpan<byte> word) =>
        word.SequenceEqual("and"u8) || word.SequenceEqual("but"u8) || word.SequenceEqual("then"u8) || word.SequenceEqual("also"u8) ||
        word.SequenceEqual("plus"u8) || word.SequenceEqual("or"u8) || word.SequenceEqual("while"u8);

    static bool ThousandsGroup(ReadOnlySpan<byte> src, int at) =>
        at + 3 <= src.Length && IsDigit(src[at]) && IsDigit(src[at + 1]) && IsDigit(src[at + 2]) && (at + 3 == src.Length || !IsDigit(src[at + 3]));

    public static bool IsDigit(byte c) => c >= (byte)'0' && c <= (byte)'9';

    public static bool IsLetter(byte c) => (c >= (byte)'a' && c <= (byte)'z') || (c >= (byte)'A' && c <= (byte)'Z') || c >= 0x80;

    public ReadOnlySpan<byte> Word(int index) => new(Norm, WordStart[index], WordEnd[index] - WordStart[index]);

    public ReadOnlySpan<byte> SourceSpan(int firstWord, int lastWord)
    {
        var start = Origin[WordStart[firstWord]];
        var end = Origin[WordEnd[lastWord] - 1] + 1;
        while (end < SourceLength && Source[end] >= 0x80 && (Source[end] & 0xC0) == 0x80) end++;
        return new ReadOnlySpan<byte>(Source, start, end - start);
    }

    public int FindPhrase(ReadOnlySpan<byte> phrase, int clause = -1, int fromWord = 0)
    {
        if (phrase.Length == 0) return -1;
        for (var w = fromWord; w < Words; w++)
        {
            if (clause >= 0 && WordClause[w] != clause) continue;
            var start = WordStart[w];
            if (start + phrase.Length > Length) return -1;
            if (!new ReadOnlySpan<byte>(Norm, start, phrase.Length).SequenceEqual(phrase)) continue;
            var end = start + phrase.Length;
            if (end == Length || Norm[end] == (byte)' ') return w;
        }
        return -1;
    }

    public int CountPhrase(ReadOnlySpan<byte> phrase, int clause = -1)
    {
        var count = 0;
        for (var w = FindPhrase(phrase, clause); w >= 0; w = FindPhrase(phrase, clause, w + 1)) count++;
        return count;
    }

    public bool HasWord(ReadOnlySpan<byte> word, int clause = -1) => FindPhrase(word, clause) >= 0;

    /// <summary>A number stated in the turns after the preamble (the conversation, not the system text).</summary>
    public bool HasTurnNumber(double value)
    {
        for (var i = 0; i < NumberCount; i++)
            if (Math.Abs(Numbers[i] - value) < 1e-9 && Origin[WordStart[NumberWord[i]]] >= Preamble) return true;
        return false;
    }

    public bool HasNumber(double value)
    {
        for (var i = 0; i < NumberCount; i++)
            if (Math.Abs(Numbers[i] - value) < 1e-9) return true;
        return false;
    }

    void CollectNumbers()
    {
        var pending = -1.0;
        var pendingWord = -1;
        for (var w = 0; w < Words && NumberCount < MaxNumbers; w++)
        {
            var word = Word(w);
            if (TryParseDecimal(word, out var parsed))
            {
                Flush(ref pending, ref pendingWord);
                AddNumber(parsed, w);
                continue;
            }
            var small = SmallNumber(word);
            if (small >= 0)
            {
                if (pending >= 20 && pending % 10 == 0 && small < 10 && pendingWord == w - 1) { pending += small; pendingWord = w; }
                else { Flush(ref pending, ref pendingWord); pending = small; pendingWord = w; }
                continue;
            }
            if (word.SequenceEqual("hundred"u8) && pendingWord == w - 1 && pending > 0) { pending *= 100; pendingWord = w; continue; }
            if (word.SequenceEqual("thousand"u8) && pendingWord == w - 1 && pending > 0) { pending *= 1000; pendingWord = w; continue; }
            if (word.SequenceEqual("dozen"u8)) { Flush(ref pending, ref pendingWord); AddNumber(12, w); continue; }
            Flush(ref pending, ref pendingWord);
        }
        Flush(ref pending, ref pendingWord);
    }

    void Flush(ref double pending, ref int pendingWord)
    {
        if (pending >= 0 && pendingWord >= 0) AddNumber(pending, pendingWord);
        pending = -1;
        pendingWord = -1;
    }

    void AddNumber(double value, int word)
    {
        if (NumberCount >= MaxNumbers) return;
        Numbers[NumberCount] = value;
        NumberWord[NumberCount] = word;
        NumberCount++;
    }

    public static bool TryParseDecimal(ReadOnlySpan<byte> word, out double value)
    {
        value = 0;
        if (word.Length == 0 || !IsDigit(word[0])) return false;
        var fraction = 0.0;
        var scale = 1.0;
        var dot = false;
        foreach (var c in word)
        {
            if (c == (byte)'.')
            {
                if (dot) return false;
                dot = true;
                continue;
            }
            if (!IsDigit(c)) return false;
            if (dot) { scale /= 10; fraction += (c - '0') * scale; }
            else value = value * 10 + (c - '0');
        }
        value += fraction;
        return true;
    }

    static int SmallNumber(ReadOnlySpan<byte> w)
    {
        if (w.SequenceEqual("zero"u8)) return 0;
        if (w.SequenceEqual("one"u8)) return 1;
        if (w.SequenceEqual("two"u8)) return 2;
        if (w.SequenceEqual("three"u8)) return 3;
        if (w.SequenceEqual("four"u8)) return 4;
        if (w.SequenceEqual("five"u8)) return 5;
        if (w.SequenceEqual("six"u8)) return 6;
        if (w.SequenceEqual("seven"u8)) return 7;
        if (w.SequenceEqual("eight"u8)) return 8;
        if (w.SequenceEqual("nine"u8)) return 9;
        if (w.SequenceEqual("ten"u8)) return 10;
        if (w.SequenceEqual("eleven"u8)) return 11;
        if (w.SequenceEqual("twelve"u8)) return 12;
        if (w.SequenceEqual("thirteen"u8)) return 13;
        if (w.SequenceEqual("fourteen"u8)) return 14;
        if (w.SequenceEqual("fifteen"u8)) return 15;
        if (w.SequenceEqual("sixteen"u8)) return 16;
        if (w.SequenceEqual("seventeen"u8)) return 17;
        if (w.SequenceEqual("eighteen"u8)) return 18;
        if (w.SequenceEqual("nineteen"u8)) return 19;
        if (w.SequenceEqual("twenty"u8)) return 20;
        if (w.SequenceEqual("thirty"u8)) return 30;
        if (w.SequenceEqual("forty"u8)) return 40;
        if (w.SequenceEqual("fifty"u8)) return 50;
        if (w.SequenceEqual("sixty"u8)) return 60;
        if (w.SequenceEqual("seventy"u8)) return 70;
        if (w.SequenceEqual("eighty"u8)) return 80;
        if (w.SequenceEqual("ninety"u8)) return 90;
        return -1;
    }

    public static int Normalize(ReadOnlySpan<byte> text, Span<byte> destination)
    {
        var length = 0;
        var space = false;
        for (var i = 0; i < text.Length; i++)
        {
            var c = text[i];
            if (c == 0xE2 && i + 2 < text.Length && text[i + 1] == 0x80)
            {
                if (text[i + 2] == 0x99 && length > 0 && i + 3 < text.Length && IsLetter(text[i + 3])) { i += 2; continue; }
                i += 2;
                space = length > 0;
                continue;
            }
            byte mapped;
            if (c >= (byte)'A' && c <= (byte)'Z') mapped = (byte)(c | 0x20);
            else if ((c >= (byte)'a' && c <= (byte)'z') || (c >= (byte)'0' && c <= (byte)'9') || c >= 0x80) mapped = c;
            else if (c == (byte)'\'' && length > 0 && i + 1 < text.Length && IsLetter(text[i + 1])) continue;
            else if (c == (byte)'.' && length > 0 && IsDigit(text[i - 1]) && i + 1 < text.Length && IsDigit(text[i + 1])) mapped = c;
            else if (c == (byte)',' && length > 0 && IsDigit(text[i - 1]) && ThousandsGroup(text, i + 1)) continue;
            else if (c == (byte)'\\' && i + 1 < text.Length)
            {
                i++;
                if (text[i] == (byte)'u') i = Math.Min(text.Length - 1, i + 4);
                space = length > 0;
                continue;
            }
            else
            {
                space = length > 0;
                continue;
            }
            if (space && length < destination.Length) destination[length++] = (byte)' ';
            space = false;
            if (length < destination.Length) destination[length++] = mapped;
        }
        return length;
    }
}
