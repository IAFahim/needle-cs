using System;
using System.Buffers.Text;

namespace Needle;

[Flags]
internal enum RepairOutcome : byte
{
    None = 0,
    Repaired = 1,
    Negated = 2,
    Invalid = 4,
    Ungrounded = 8,
    Duplicate = 16,
}

internal enum ValueKind : byte { String, Number, True, False, Null, Object, Array }

internal struct CallSlot
{
    public int Tool;
    public int FirstArg;
    public int ArgCount;
    public int Anchor;
    public bool Removed;
}

internal struct ArgSlot
{
    public int Call;
    public int Param;
    public int KeyStart;
    public int KeyLength;
    public int ValueStart;
    public int ValueLength;
    public ValueKind Kind;
    public bool Removed;
    public bool FromArena;
    public bool FromDefault;
}

internal sealed class RepairScratch
{
    public const int MaxClauses = 32;
    public const int MaxQuotes = 8;
    public const int MaxPhones = 8;

    public readonly CallSlot[] Calls;
    public readonly ArgSlot[] Args;
    public readonly byte[] Arena;
    public readonly byte[] Output;
    public readonly byte[] Words;
    public readonly int[] ClauseScore;
    public readonly int[] QuoteStart;
    public readonly int[] QuoteLength;
    public readonly int[] PhoneStart;
    public readonly int[] PhoneLength;
    public readonly byte[] Ungrounded;
    public int UngroundedLength;
    public int ArgCount;
    public int CallCount;
    public readonly CallSpan[] Spans;
    public int ArenaLength;
    public byte[] PhoneText = Array.Empty<byte>();
    public readonly RepairText Value = new(1024);

    public RepairScratch(int maxCalls, int maxArgs, int callBytes)
    {
        Calls = new CallSlot[Math.Max(1, maxCalls)];
        Spans = new CallSpan[Math.Max(1, maxCalls)];
        Args = new ArgSlot[Math.Max(8, maxArgs)];
        Arena = new byte[Math.Max(4096, callBytes * 2)];
        Output = new byte[Math.Max(4096, callBytes * 2 + 1024)];
        Words = new byte[Math.Max(1024, callBytes)];
        ClauseScore = new int[MaxClauses];
        QuoteStart = new int[MaxQuotes];
        QuoteLength = new int[MaxQuotes];
        PhoneStart = new int[MaxPhones];
        PhoneLength = new int[MaxPhones];
        Ungrounded = new byte[2048];
    }
}

internal static class ArgumentRepair
{
    public static RepairOutcome Apply(RepairPlan plan, RepairScratch s, RepairText request, RepairText history, DateTime? today,
        Span<byte> calls, int length, Span<CallSpan> spans, ref int callCount, out int required)
    {
        required = length;
        s.UngroundedLength = 0;
        s.ArgCount = 0;
        s.ArenaLength = 0;
        if (callCount == 0 || callCount > s.Calls.Length) return RepairOutcome.None;
        s.CallCount = callCount;
        var outcome = RepairOutcome.None;
        var source = calls[..length];
        for (var c = 0; c < callCount; c++)
        {
            ref var call = ref s.Calls[c];
            call.Tool = spans[c].Tool;
            call.FirstArg = s.ArgCount;
            call.Removed = false;
            if (call.Tool < 0 || call.Tool >= plan.Tools.Length || !ParseArguments(plan.Tools[call.Tool], s, c, source, spans[c].ArgumentsStart, spans[c].ArgumentsLength)) return RepairOutcome.None;
            call.ArgCount = s.ArgCount - call.FirstArg;
        }
        FindQuotes(request, s);
        FindPhones(request, s);
        for (var c = 0; c < callCount; c++)
        {
            ref var call = ref s.Calls[c];
            call.Anchor = Anchor(plan, s, source, request, c);
            var tool = plan.Tools[call.Tool];
            if (Negated(request, s, call.Anchor))
            {
                call.Removed = true;
                outcome |= RepairOutcome.Negated;
                continue;
            }
            if (tool.Paired >= 0)
            {
                var direction = Cue(request, tool.Family, call.Anchor);
                if (direction != 0 && direction == -tool.Direction)
                {
                    call.Tool = tool.Paired;
                    RebindParams(plan.Tools[call.Tool], s, c, source);
                    outcome |= RepairOutcome.Repaired;
                }
            }
            if (RepairCall(plan.Tools[call.Tool], s, source, request, history, today, c)) outcome |= RepairOutcome.Repaired;
            if (!InRange(plan.Tools[call.Tool], s, source, c))
            {
                call.Removed = true;
                outcome |= RepairOutcome.Invalid;
            }
        }
        var written = Write(plan, s, source, callCount, out var kept, ref outcome);
        if (written < 0) return RepairOutcome.None;
        if (Ungrounded(plan, s, history, request, today, kept)) outcome |= RepairOutcome.Ungrounded;
        if ((outcome & (RepairOutcome.Repaired | RepairOutcome.Negated | RepairOutcome.Invalid | RepairOutcome.Duplicate)) == 0) return outcome;
        required = written;
        // Calls, spans and count change together or not at all, so a caller never sees spans into bytes it does not hold.
        if (written > calls.Length) return outcome & RepairOutcome.Ungrounded;
        s.Output.AsSpan(0, written).CopyTo(calls);
        s.Spans.AsSpan(0, kept).CopyTo(spans);
        callCount = kept;
        return outcome;
    }

    static bool ParseArguments(RepairTool tool, RepairScratch s, int call, ReadOnlySpan<byte> json, int start, int length)
    {
        var end = start + length;
        var at = start;
        if (at >= end || json[at] != (byte)'{') return false;
        at++;
        if (at < end && json[at] == (byte)'}') return true;
        while (at < end)
        {
            if (json[at] != (byte)'"') return false;
            var keyStart = at + 1;
            if (!SkipString(json, ref at, end)) return false;
            var keyLength = at - 1 - keyStart;
            if (at >= end || json[at] != (byte)':') return false;
            at++;
            var valueStart = at;
            if (!SkipValue(json, ref at, end, out var kind)) return false;
            if (s.ArgCount >= s.Args.Length) return false;
            s.Args[s.ArgCount++] = new ArgSlot
            {
                Call = call,
                Param = tool.FindParam(json.Slice(keyStart, keyLength)),
                KeyStart = keyStart,
                KeyLength = keyLength,
                ValueStart = valueStart,
                ValueLength = at - valueStart,
                Kind = kind,
            };
            if (at < end && json[at] == (byte)',') { at++; continue; }
            if (at < end && json[at] == (byte)'}') return at + 1 == end;
            return false;
        }
        return false;
    }

    static void RebindParams(RepairTool tool, RepairScratch s, int call, ReadOnlySpan<byte> json)
    {
        for (var a = s.Calls[call].FirstArg; a < s.Calls[call].FirstArg + s.Calls[call].ArgCount; a++)
            s.Args[a].Param = s.Args[a].KeyStart >= 0 ? tool.FindParam(json.Slice(s.Args[a].KeyStart, s.Args[a].KeyLength)) : -1;
    }

    static bool SkipString(ReadOnlySpan<byte> json, ref int at, int end)
    {
        at++;
        while (at < end)
        {
            var c = json[at++];
            if (c == (byte)'\\') { at++; continue; }
            if (c == (byte)'"') return true;
        }
        return false;
    }

    static bool SkipValue(ReadOnlySpan<byte> json, ref int at, int end, out ValueKind kind)
    {
        kind = ValueKind.Null;
        if (at >= end) return false;
        var c = json[at];
        if (c == (byte)'"') { kind = ValueKind.String; return SkipString(json, ref at, end); }
        if (c is (byte)'{' or (byte)'[')
        {
            kind = c == (byte)'{' ? ValueKind.Object : ValueKind.Array;
            var depth = 0;
            while (at < end)
            {
                var b = json[at];
                if (b == (byte)'"') { if (!SkipString(json, ref at, end)) return false; continue; }
                if (b is (byte)'{' or (byte)'[') depth++;
                else if (b is (byte)'}' or (byte)']') depth--;
                at++;
                if (depth == 0) return true;
            }
            return false;
        }
        var start = at;
        while (at < end && json[at] is not ((byte)',' or (byte)'}' or (byte)']')) at++;
        var literal = json.Slice(start, at - start);
        kind = literal.SequenceEqual("true"u8) ? ValueKind.True : literal.SequenceEqual("false"u8) ? ValueKind.False : literal.SequenceEqual("null"u8) ? ValueKind.Null : ValueKind.Number;
        return literal.Length > 0;
    }

    static ReadOnlySpan<byte> Value(RepairScratch s, ReadOnlySpan<byte> json, in ArgSlot arg) =>
        arg.FromArena ? new ReadOnlySpan<byte>(s.Arena, arg.ValueStart, arg.ValueLength) : json.Slice(arg.ValueStart, arg.ValueLength);

    static ReadOnlySpan<byte> StringContent(RepairScratch s, ReadOnlySpan<byte> json, in ArgSlot arg)
    {
        var value = Value(s, json, arg);
        return arg.Kind == ValueKind.String && value.Length >= 2 ? value[1..^1] : default;
    }

    static int Normalized(RepairScratch s, ReadOnlySpan<byte> text, int offset = 0) =>
        RepairText.Normalize(text, s.Words.AsSpan(offset));

    static int Anchor(RepairPlan plan, RepairScratch s, ReadOnlySpan<byte> json, RepairText request, int call)
    {
        if (request.Clauses <= 1) return request.Clauses - 1;
        var clauses = Math.Min(request.Clauses, RepairScratch.MaxClauses);
        Array.Clear(s.ClauseScore, 0, clauses);
        var slot = s.Calls[call];
        var tool = plan.Tools[slot.Tool];
        foreach (var word in tool.NameWords) Score(request, s, word, 1);
        for (var a = slot.FirstArg; a < slot.FirstArg + slot.ArgCount; a++)
        {
            ref var arg = ref s.Args[a];
            if (arg.Kind == ValueKind.String)
            {
                var length = Normalized(s, StringContent(s, json, arg));
                ScoreWords(request, s, s.Words.AsSpan(0, length), 2);
            }
            else if (arg.Kind == ValueKind.Number && Utf8Parser.TryParse(Value(s, json, arg), out double number, out _))
                for (var n = 0; n < request.NumberCount; n++)
                    if (Math.Abs(request.Numbers[n] - number) < 1e-9 && request.WordClause[request.NumberWord[n]] < clauses) s.ClauseScore[request.WordClause[request.NumberWord[n]]] += 2;
        }
        var best = -1;
        var bestScore = 0;
        var tie = false;
        for (var k = 0; k < clauses; k++)
        {
            if (s.ClauseScore[k] > bestScore) { best = k; bestScore = s.ClauseScore[k]; tie = false; }
            else if (s.ClauseScore[k] == bestScore && bestScore > 0) tie = true;
        }
        return tie ? -1 : best;
    }

    static void ScoreWords(RepairText request, RepairScratch s, ReadOnlySpan<byte> words, int weight)
    {
        var start = 0;
        for (var i = 0; i <= words.Length; i++)
        {
            if (i < words.Length && words[i] != (byte)' ') continue;
            if (i - start >= 3 && !IsStopWord(words[start..i])) Score(request, s, words[start..i], weight);
            start = i + 1;
        }
    }

    static void Score(RepairText request, RepairScratch s, ReadOnlySpan<byte> word, int weight)
    {
        for (var w = request.FindPhrase(word); w >= 0; w = request.FindPhrase(word, -1, w + 1))
            if (request.WordClause[w] < RepairScratch.MaxClauses) s.ClauseScore[request.WordClause[w]] += weight;
    }

    static bool IsStopWord(ReadOnlySpan<byte> w) =>
        w.SequenceEqual("the"u8) || w.SequenceEqual("and"u8) || w.SequenceEqual("for"u8) || w.SequenceEqual("with"u8) || w.SequenceEqual("from"u8) ||
        w.SequenceEqual("that"u8) || w.SequenceEqual("this"u8) || w.SequenceEqual("please"u8);

    static bool Negated(RepairText request, RepairScratch s, int clause)
    {
        if (clause < 0) return false;
        var negated = false;
        for (var w = 0; w < request.Words; w++)
        {
            if (request.WordClause[w] != clause) continue;
            var word = request.Word(w);
            // A negation inside the content of the request ("text Sam that I won't make it", "remind me to never ...") is not a refusal.
            if (Complement(word)) break;
            if (InQuote(request, s, w)) continue;
            var next = w + 1 < request.Words && request.WordClause[w + 1] == clause ? request.Word(w + 1) : default;
            var after = w + 2 < request.Words && request.WordClause[w + 2] == clause ? request.Word(w + 2) : default;
            if (word.SequenceEqual("dont"u8) || word.SequenceEqual("not"u8) || word.SequenceEqual("never"u8) || word.SequenceEqual("without"u8) ||
                word.SequenceEqual("except"u8) || word.SequenceEqual("neither"u8) || word.SequenceEqual("nor"u8) || word.SequenceEqual("wont"u8) ||
                word.SequenceEqual("shouldnt"u8) || word.SequenceEqual("no"u8) || word.SequenceEqual("avoid"u8))
            {
                if (next.SequenceEqual("forget"u8) || next.SequenceEqual("disturb"u8) || next.SequenceEqual("matter"u8) || next.SequenceEqual("only"u8) ||
                    next.SequenceEqual("problem"u8) || next.SequenceEqual("longer"u8) || next.SequenceEqual("later"u8) || next.SequenceEqual("more"u8) && after.SequenceEqual("than"u8) ||
                    word.SequenceEqual("not"u8) && w > 0 && request.Word(w - 1).SequenceEqual("do"u8) && next.SequenceEqual("disturb"u8))
                    continue;
                if (word.SequenceEqual("no"u8) && next.Length == 0) continue;
                negated = true;
            }
        }
        return negated;
    }

    static bool Complement(ReadOnlySpan<byte> word) =>
        word.SequenceEqual("that"u8) || word.SequenceEqual("saying"u8) || word.SequenceEqual("say"u8) || word.SequenceEqual("says"u8) ||
        word.SequenceEqual("to"u8) || word.SequenceEqual("about"u8) || word.SequenceEqual("titled"u8) || word.SequenceEqual("called"u8);

    static bool InQuote(RepairText request, RepairScratch s, int word)
    {
        var at = request.Origin[request.WordStart[word]];
        for (var q = 0; q < RepairScratch.MaxQuotes && s.QuoteLength[q] > 0; q++)
            if (at >= s.QuoteStart[q] && at < s.QuoteStart[q] + s.QuoteLength[q]) return true;
        return false;
    }

    static int Wide(RepairScratch s, int anchor) => anchor >= 0 && s.CallCount == 1 ? -1 : -2;

    static sbyte Cue(RepairText request, Polar family, int clause)
    {
        if (family == Polar.None) return 0;
        var positive = 0;
        var negative = 0;
        for (var v = 0; v < RepairPlan.CueWords.Length; v++)
        {
            if (RepairPlan.CueFamily[v] != family) continue;
            var cue = RepairPlan.CueWords[v];
            for (var w = request.FindPhrase(cue, clause); w >= 0; w = request.FindPhrase(cue, clause, w + 1))
            {
                if (cue.AsSpan().SequenceEqual("on"u8) && !SwitchVerbBefore(request, w)) continue;
                if (RepairPlan.CueDirection[v] > 0) positive++;
                else negative++;
            }
        }
        if (positive > 0 && negative == 0) return 1;
        if (negative > 0 && positive == 0) return -1;
        return 0;
    }

    static bool SwitchVerbBefore(RepairText request, int word)
    {
        var clause = request.WordClause[word];
        for (var w = word - 1; w >= 0 && request.WordClause[w] == clause; w--)
        {
            var v = request.Word(w);
            if (v.SequenceEqual("turn"u8) || v.SequenceEqual("switch"u8) || v.SequenceEqual("put"u8) || v.SequenceEqual("power"u8) || v.SequenceEqual("flip"u8) ||
                v.SequenceEqual("run"u8) || v.SequenceEqual("keep"u8) || v.SequenceEqual("leave"u8) || v.SequenceEqual("set"u8) || v.SequenceEqual("lights"u8) ||
                v.SequenceEqual("light"u8) || v.SequenceEqual("turned"u8) || v.SequenceEqual("switched"u8))
                return true;
        }
        return false;
    }

    static bool RepairCall(RepairTool tool, RepairScratch s, ReadOnlySpan<byte> json, RepairText request, RepairText history, DateTime? today, int call)
    {
        var changed = false;
        ref var slot = ref s.Calls[call];
        var anchor = slot.Anchor;
        for (var a = slot.FirstArg; a < slot.FirstArg + slot.ArgCount; a++)
        {
            ref var arg = ref s.Args[a];
            if (arg.Param < 0 || arg.Removed) continue;
            var param = tool.Params[arg.Param];
            if (param.Kind == ParamKind.Boolean && arg.Kind is ValueKind.True or ValueKind.False && param.BoolPolar != Polar.None)
            {
                var direction = Cue(request, param.BoolPolar, anchor);
                if (direction == 0 && Wide(s, anchor) == -1) direction = Cue(request, param.BoolPolar, -1);
                var want = direction > 0 ? ValueKind.True : ValueKind.False;
                if (direction != 0 && arg.Kind != want) { SetLiteral(s, ref arg, want); changed = true; }
                continue;
            }
            if (param.Options.Length > 0)
            {
                changed |= RepairEnum(param, s, json, request, history, ref arg, anchor);
                continue;
            }
            if (param.Kind == ParamKind.String && arg.Kind == ValueKind.String)
            {
                if (param.Date && today.HasValue && tool.DateParams == 1) changed |= RepairDate(s, json, request, today.Value, ref arg, anchor);
                else if (param.Phone) changed |= RepairPhone(s, json, ref arg);
                else changed |= RepairQuoted(s, json, request, ref arg);
            }
        }
        changed |= RepairNames(tool, s, json, request, call);
        changed |= RepairRoute(tool, s, json, request, call, anchor);
        changed |= RepairBy(tool, s, json, request, call, anchor);
        for (var a = slot.FirstArg; a < slot.FirstArg + slot.ArgCount; a++)
        {
            ref var arg = ref s.Args[a];
            if (arg.Param < 0 || arg.Removed) continue;
            var param = tool.Params[arg.Param];
            if (param.Kind is ParamKind.Integer or ParamKind.Number && arg.Kind == ValueKind.Number && param.Options.Length == 0)
            {
                var stated = Utf8Parser.TryParse(Value(s, json, arg), out double number, out _) && history.HasNumber(number);
                if (stated || arg.FromDefault) continue;
                if (param.Required)
                {
                    if (param.DefaultJson != null && !AnyNumber(request, anchor)) { SetJson(s, ref arg, param.DefaultJson, ValueKind.Number); arg.FromDefault = true; changed = true; }
                }
                else { arg.Removed = true; changed = true; }
                continue;
            }
            if (param.Kind == ParamKind.String && arg.Kind == ValueKind.String && !param.Date && arg.Param != tool.LastName)
            {
                var content = StringContent(s, json, arg);
                if (!param.Required && content.Length == 0) { arg.Removed = true; changed = true; continue; }
                if (arg.FromArena || arg.FromDefault || Stated(s, history, content, param.Phone)) continue;
                if (!param.Required) { arg.Removed = true; changed = true; }
                else if (param.DefaultJson != null) { SetJson(s, ref arg, param.DefaultJson, ValueKind.String); arg.FromDefault = true; changed = true; }
            }
        }
        changed |= FillMissing(tool, s, json, request, call, anchor);
        return changed;
    }

    static bool RepairEnum(RepairParam param, RepairScratch s, ReadOnlySpan<byte> json, RepairText request, RepairText history, ref ArgSlot arg, int anchor)
    {
        var current = CurrentOption(param, s, json, arg);
        var named = NamedOption(param, request, anchor);
        if (named < 0 && Wide(s, anchor) == -1) named = NamedOption(param, request, -1);
        if (named >= 0)
        {
            if (current != named && (current < 0 || request.FindPhrase(param.OptionWords[current], anchor) < 0))
            {
                SetJson(s, ref arg, param.Options[named], param.Options[named][0] == (byte)'"' ? ValueKind.String : ValueKind.Number);
                return true;
            }
            return false;
        }
        if (param.OptionsPolar != Polar.None)
        {
            var direction = Cue(request, param.OptionsPolar, anchor);
            if (direction == 0 && Wide(s, anchor) == -1) direction = Cue(request, param.OptionsPolar, -1);
            if (direction != 0)
                for (var o = 0; o < param.Options.Length; o++)
                    if (param.OptionFamily[o] == param.OptionsPolar && param.OptionDirection[o] == direction)
                    {
                        if (o == current) return false;
                        SetJson(s, ref arg, param.Options[o], ValueKind.String);
                        return true;
                    }
        }
        if (!param.Required && current >= 0 && param.OptionWords[current].Length > 0 && !Stated(s, history, param.OptionWords[current], false))
        {
            arg.Removed = true;
            return true;
        }
        return false;
    }

    static int CurrentOption(RepairParam param, RepairScratch s, ReadOnlySpan<byte> json, in ArgSlot arg)
    {
        var value = Value(s, json, arg);
        for (var o = 0; o < param.Options.Length; o++)
            if (value.SequenceEqual(param.Options[o])) return o;
        return -1;
    }

    static int NamedOption(RepairParam param, RepairText request, int clause)
    {
        var found = -1;
        for (var o = 0; o < param.Options.Length; o++)
        {
            if (param.OptionWords[o].Length == 0 || request.FindPhrase(param.OptionWords[o], clause) < 0) continue;
            if (param.OptionFamily[o] == Polar.Power && param.OptionWords[o].AsSpan().SequenceEqual("on"u8))
            {
                var w = request.FindPhrase("on"u8, clause);
                var verb = false;
                for (; w >= 0; w = request.FindPhrase("on"u8, clause, w + 1)) verb |= SwitchVerbBefore(request, w);
                if (!verb) continue;
            }
            if (found >= 0 && !Contains(param.OptionWords[o], param.OptionWords[found]) && !Contains(param.OptionWords[found], param.OptionWords[o])) return -1;
            if (found < 0 || param.OptionWords[o].Length > param.OptionWords[found].Length) found = o;
        }
        return found;
    }

    static bool Contains(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle) => haystack.IndexOf(needle) >= 0;

    static bool Stated(RepairScratch s, RepairText history, ReadOnlySpan<byte> raw, bool digits)
    {
        if (digits)
        {
            var count = Digits(raw, s.Words);
            if (count == 0) return false;
            var histDigits = Digits(new ReadOnlySpan<byte>(history.Source, 0, history.SourceLength), s.Words.AsSpan(count));
            return s.Words.AsSpan(count, histDigits).IndexOf(s.Words.AsSpan(0, count)) >= 0;
        }
        var length = Normalized(s, raw);
        if (length == 0) return true;
        var words = s.Words.AsSpan(0, length);
        var start = 0;
        var anyLong = false;
        for (var i = 0; i <= words.Length; i++)
        {
            if (i < words.Length && words[i] != (byte)' ') continue;
            var word = words[start..i];
            if (word.Length >= 3 && !IsStopWord(word))
            {
                anyLong = true;
                if (history.HasWord(word) || Stem(history, word)) return true;
            }
            start = i + 1;
        }
        return !anyLong && history.FindPhrase(words) >= 0;
    }

    static bool Stem(RepairText history, ReadOnlySpan<byte> word)
    {
        if (word.Length >= 4 && word[^1] == (byte)'s' && history.HasWord(word[..^1])) return true;
        for (var w = 0; w < history.Words; w++)
        {
            var h = history.Word(w);
            if (h.Length >= 4 && h.Length == word.Length + 1 && h[^1] == (byte)'s' && h[..^1].SequenceEqual(word)) return true;
            if (word.Length >= 5 && h.Length >= 5 && h[..5].SequenceEqual(word[..5])) return true;
        }
        return false;
    }

    static int Digits(ReadOnlySpan<byte> text, Span<byte> destination)
    {
        var n = 0;
        foreach (var c in text)
            if (RepairText.IsDigit(c) && n < destination.Length) destination[n++] = c;
        return n;
    }

    static bool AnyNumber(RepairText request, int clause)
    {
        for (var n = 0; n < request.NumberCount; n++)
            if (clause < 0 || request.WordClause[request.NumberWord[n]] == clause) return true;
        return false;
    }

    static void FindQuotes(RepairText request, RepairScratch s)
    {
        var count = 0;
        var text = new ReadOnlySpan<byte>(request.Source, 0, request.SourceLength);
        for (var i = 0; i < text.Length && count < RepairScratch.MaxQuotes; i++)
        {
            int open, close;
            if (text[i] == (byte)'"') { open = 1; close = text[(i + 1)..].IndexOf((byte)'"'); }
            else if (i + 2 < text.Length && text[i] == 0xE2 && text[i + 1] == 0x80 && text[i + 2] == 0x9C) { open = 3; close = text[(i + 3)..].IndexOf("”"u8); }
            else if (text[i] == (byte)'\'' && (i == 0 || text[i - 1] == (byte)' ')) { open = 1; close = SingleClose(text, i + 1); }
            else continue;
            if (close <= 0) continue;
            s.QuoteStart[count] = i + open;
            s.QuoteLength[count] = close;
            count++;
            i += open + close;
        }
        for (var q = count; q < RepairScratch.MaxQuotes; q++) s.QuoteLength[q] = 0;
    }

    static int SingleClose(ReadOnlySpan<byte> text, int from)
    {
        for (var j = from; j < text.Length; j++)
            if (text[j] == (byte)'\'' && (j + 1 == text.Length || text[j + 1] is (byte)' ' or (byte)'.' or (byte)',' or (byte)'!' or (byte)'?' or (byte)';'))
                return j - from;
        return -1;
    }

    static bool RepairQuoted(RepairScratch s, ReadOnlySpan<byte> json, RepairText request, ref ArgSlot arg)
    {
        var content = StringContent(s, json, arg);
        var valueLength = Normalized(s, content);
        if (valueLength == 0) return false;
        var match = -1;
        for (var q = 0; q < RepairScratch.MaxQuotes && s.QuoteLength[q] > 0; q++)
        {
            var quote = new ReadOnlySpan<byte>(request.Source, s.QuoteStart[q], s.QuoteLength[q]);
            var quoteLength = Normalized(s, quote, valueLength + 1);
            if (quoteLength == 0) continue;
            var v = s.Words.AsSpan(0, valueLength);
            var w = s.Words.AsSpan(valueLength + 1, quoteLength);
            if (!(Contains(w, v) || Contains(v, w))) continue;
            if (match >= 0) return false;
            match = q;
        }
        if (match < 0) return false;
        var verbatim = new ReadOnlySpan<byte>(request.Source, s.QuoteStart[match], s.QuoteLength[match]);
        if (verbatim.SequenceEqual(content)) return false;
        return SetString(s, ref arg, verbatim);
    }

    static void FindPhones(RepairText request, RepairScratch s)
    {
        var count = 0;
        s.PhoneText = request.Source;
        var text = new ReadOnlySpan<byte>(request.Source, 0, request.SourceLength);
        var i = 0;
        while (i < text.Length && count < RepairScratch.MaxPhones)
        {
            if (!(RepairText.IsDigit(text[i]) || text[i] == (byte)'+' || text[i] == (byte)'(')) { i++; continue; }
            var start = i;
            var digits = 0;
            var last = i;
            while (i < text.Length && (RepairText.IsDigit(text[i]) || text[i] is (byte)' ' or (byte)'-' or (byte)'.' or (byte)'(' or (byte)')' or (byte)'+'))
            {
                if (RepairText.IsDigit(text[i])) { digits++; last = i; }
                i++;
            }
            if (digits >= 7 && digits <= 15)
            {
                s.PhoneStart[count] = start;
                s.PhoneLength[count] = last - start + 1;
                count++;
            }
        }
        for (var p = count; p < RepairScratch.MaxPhones; p++) s.PhoneLength[p] = 0;
    }

    static bool RepairPhone(RepairScratch s, ReadOnlySpan<byte> json, ref ArgSlot arg)
    {
        var content = StringContent(s, json, arg);
        var have = Digits(content, s.Words);
        if (have < 3) return false;
        var match = -1;
        for (var p = 0; p < RepairScratch.MaxPhones && s.PhoneLength[p] > 0; p++)
        {
            var candidate = DigitsOf(s, p, have);
            if (candidate <= have) continue;
            if (s.Words.AsSpan(have, candidate).IndexOf(s.Words.AsSpan(0, have)) < 0) continue;
            if (match >= 0) return false;
            match = p;
        }
        return match >= 0 && SetString(s, ref arg, PhoneSpan(s, match));
    }

    static int DigitsOf(RepairScratch s, int phone, int offset) => Digits(PhoneSpan(s, phone), s.Words.AsSpan(offset));

    static ReadOnlySpan<byte> PhoneSpan(RepairScratch s, int phone) => new(s.PhoneText, s.PhoneStart[phone], s.PhoneLength[phone]);

    static bool RepairDate(RepairScratch s, ReadOnlySpan<byte> json, RepairText request, DateTime today, ref ArgSlot arg, int anchor)
    {
        var content = StringContent(s, json, arg);
        if (content.Length < 10 || !IsIsoDate(content)) return false;
        if (!ResolveDate(request, today, anchor, out var date) && !(Wide(s, anchor) == -1 && ResolveDate(request, today, -1, out date))) return false;
        Span<byte> iso = stackalloc byte[10];
        WriteIso(iso, date);
        if (content[..10].SequenceEqual(iso)) return false;
        var start = s.ArenaLength;
        if (start + content.Length + 2 > s.Arena.Length) return false;
        s.Arena[start] = (byte)'"';
        iso.CopyTo(s.Arena.AsSpan(start + 1));
        content[10..].CopyTo(s.Arena.AsSpan(start + 11));
        s.Arena[start + 1 + content.Length] = (byte)'"';
        s.ArenaLength += content.Length + 2;
        arg.ValueStart = start;
        arg.ValueLength = content.Length + 2;
        arg.FromArena = true;
        return true;
    }

    static bool IsIsoDate(ReadOnlySpan<byte> v) =>
        RepairText.IsDigit(v[0]) && RepairText.IsDigit(v[1]) && RepairText.IsDigit(v[2]) && RepairText.IsDigit(v[3]) && v[4] == (byte)'-' &&
        RepairText.IsDigit(v[5]) && RepairText.IsDigit(v[6]) && v[7] == (byte)'-' && RepairText.IsDigit(v[8]) && RepairText.IsDigit(v[9]);

    static void WriteIso(Span<byte> destination, DateTime date)
    {
        var y = date.Year;
        destination[0] = (byte)('0' + y / 1000 % 10);
        destination[1] = (byte)('0' + y / 100 % 10);
        destination[2] = (byte)('0' + y / 10 % 10);
        destination[3] = (byte)('0' + y % 10);
        destination[4] = (byte)'-';
        destination[5] = (byte)('0' + date.Month / 10);
        destination[6] = (byte)('0' + date.Month % 10);
        destination[7] = (byte)'-';
        destination[8] = (byte)('0' + date.Day / 10);
        destination[9] = (byte)('0' + date.Day % 10);
    }

    public static bool ResolveDate(RepairText request, DateTime today, int clause, out DateTime date)
    {
        date = today;
        var found = 0;
        for (var w = 0; w < request.Words; w++)
        {
            if (clause >= 0 && request.WordClause[w] != clause) continue;
            var word = request.Word(w);
            var previous = w > 0 ? request.Word(w - 1) : default;
            DateTime candidate;
            if (word.SequenceEqual("today"u8) || word.SequenceEqual("tonight"u8)) candidate = today;
            else if (word.SequenceEqual("tomorrow"u8))
            {
                var dayAfter = w >= 2 && previous.SequenceEqual("after"u8) && request.Word(w - 2).SequenceEqual("day"u8);
                candidate = today.AddDays(dayAfter ? 2 : 1);
            }
            else if (word.SequenceEqual("yesterday"u8)) candidate = today.AddDays(-1);
            else if (Weekday(word) is var weekday && weekday >= 0)
            {
                var ahead = ((weekday - (int)today.DayOfWeek) % 7 + 7) % 7;
                if (previous.SequenceEqual("next"u8) && ahead == 0) ahead = 7;
                candidate = today.AddDays(ahead);
            }
            else if (Month(word) is var month && month > 0 && w + 1 < request.Words && DayNumber(request.Word(w + 1)) is var day && day > 0)
            {
                if (!TryDate(today, month, day, w + 2 < request.Words ? request.Word(w + 2) : default, out candidate)) continue;
            }
            else if (DayNumber(word) is var dayFirst && dayFirst > 0 && w + 1 < request.Words)
            {
                var monthAt = request.Word(w + 1).SequenceEqual("of"u8) && w + 2 < request.Words ? w + 2 : w + 1;
                var monthAfter = Month(request.Word(monthAt));
                if (monthAfter <= 0 || !TryDate(today, monthAfter, dayFirst, monthAt + 1 < request.Words ? request.Word(monthAt + 1) : default, out candidate)) continue;
                w = monthAt;
            }
            else if (word.SequenceEqual("in"u8) && w + 2 < request.Words && RepairText.TryParseDecimal(request.Word(w + 1), out var amount) && amount < 400)
            {
                var unit = request.Word(w + 2);
                if (unit.SequenceEqual("days"u8) || unit.SequenceEqual("day"u8)) candidate = today.AddDays(amount);
                else if (unit.SequenceEqual("weeks"u8) || unit.SequenceEqual("week"u8)) candidate = today.AddDays(7 * amount);
                else continue;
            }
            else continue;
            if (found > 0 && candidate.Date != date.Date) return false;
            date = candidate;
            found++;
        }
        return found > 0;
    }

    static bool TryDate(DateTime today, int month, int day, ReadOnlySpan<byte> yearWord, out DateTime date)
    {
        date = default;
        var year = today.Year;
        var explicitYear = 0.0;
        var stated = yearWord.Length == 4 && RepairText.TryParseDecimal(yearWord, out explicitYear) && explicitYear >= 1900 && explicitYear <= 2200;
        if (stated) year = (int)explicitYear;
        // With no year, a month and day that already passed this year mean the next one ("on January 5" said in November).
        else if (day <= DateTime.DaysInMonth(year, month) && new DateTime(year, month, day) < today.Date) year++;
        if (day > DateTime.DaysInMonth(year, month)) return false;
        date = new DateTime(year, month, day);
        return true;
    }

    static int DayNumber(ReadOnlySpan<byte> word)
    {
        var digits = 0;
        var value = 0;
        while (digits < word.Length && RepairText.IsDigit(word[digits])) value = value * 10 + (word[digits++] - '0');
        if (digits == 0 || digits > 2) return -1;
        var rest = word[digits..];
        if (rest.Length != 0 && !rest.SequenceEqual("st"u8) && !rest.SequenceEqual("nd"u8) && !rest.SequenceEqual("rd"u8) && !rest.SequenceEqual("th"u8)) return -1;
        return value is >= 1 and <= 31 ? value : -1;
    }

    static int Weekday(ReadOnlySpan<byte> w)
    {
        if (w.SequenceEqual("sunday"u8)) return 0;
        if (w.SequenceEqual("monday"u8)) return 1;
        if (w.SequenceEqual("tuesday"u8)) return 2;
        if (w.SequenceEqual("wednesday"u8)) return 3;
        if (w.SequenceEqual("thursday"u8)) return 4;
        if (w.SequenceEqual("friday"u8)) return 5;
        if (w.SequenceEqual("saturday"u8)) return 6;
        return -1;
    }

    static int Month(ReadOnlySpan<byte> w)
    {
        if (w.Length < 3) return -1;
        if (w.SequenceEqual("may"u8)) return 5;
        ReadOnlySpan<byte> names = "januaryfebruarymarchapriljunejulyaugustseptemberoctobernovemberdecember"u8;
        ReadOnlySpan<int> starts = stackalloc int[] { 0, 7, 15, 20, 25, 29, 33, 39, 48, 55, 63 };
        ReadOnlySpan<int> lengths = stackalloc int[] { 7, 8, 5, 5, 4, 4, 6, 9, 7, 8, 8 };
        ReadOnlySpan<int> numbers = stackalloc int[] { 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12 };
        for (var m = 0; m < starts.Length; m++)
        {
            var full = names.Slice(starts[m], lengths[m]);
            if (w.SequenceEqual(full)) return numbers[m];
            if (w.Length == 3 && full[..3].SequenceEqual(w)) return numbers[m];
            if (w.Length == 4 && numbers[m] == 9 && w.SequenceEqual("sept"u8)) return 9;
        }
        return -1;
    }

    static bool RepairNames(RepairTool tool, RepairScratch s, ReadOnlySpan<byte> json, RepairText request, int call)
    {
        if (tool.FirstName < 0) return false;
        var first = FindArg(s, call, tool.FirstName);
        var last = FindArg(s, call, tool.LastName);
        var lastEmpty = last < 0 || s.Args[last].Kind != ValueKind.String || StringContent(s, json, s.Args[last]).Length == 0;
        if (first >= 0 && s.Args[first].Kind == ValueKind.String)
        {
            var content = StringContent(s, json, s.Args[first]);
            var space = content.LastIndexOf((byte)' ');
            if (space > 0 && space < content.Length - 1 && (lastEmpty || StringContent(s, json, s.Args[last]).SequenceEqual(content)))
            {
                var surname = content[(space + 1)..];
                var given = content[..space];
                var lastSlot = last >= 0 ? last : AddArg(s, call, tool.LastName);
                if (lastSlot < 0) return false;
                return SetString(s, ref s.Args[lastSlot], surname) & SetString(s, ref s.Args[first], given);
            }
            if (lastEmpty && space < 0)
            {
                var length = Normalized(s, content);
                var w = request.FindPhrase(s.Words.AsSpan(0, length));
                if (w >= 0 && length > 0 && w + 1 < request.Words && request.WordClause[w + 1] == request.WordClause[w])
                {
                    var next = request.SourceSpan(w + 1, w + 1);
                    if (next.Length > 1 && next[0] >= (byte)'A' && next[0] <= (byte)'Z' && !IsStopWord(request.Word(w + 1)))
                    {
                        var lastSlot = last >= 0 ? last : AddArg(s, call, tool.LastName);
                        return lastSlot >= 0 && SetString(s, ref s.Args[lastSlot], next);
                    }
                }
            }
        }
        else if (first < 0 && last >= 0 && s.Args[last].Kind == ValueKind.String)
        {
            var content = StringContent(s, json, s.Args[last]);
            var space = content.IndexOf((byte)' ');
            if (space > 0 && space < content.Length - 1)
            {
                var slot = AddArg(s, call, tool.FirstName);
                return slot >= 0 && SetString(s, ref s.Args[slot], content[..space]) & SetString(s, ref s.Args[last], content[(space + 1)..]);
            }
        }
        return false;
    }

    static bool RepairRoute(RepairTool tool, RepairScratch s, ReadOnlySpan<byte> json, RepairText request, int call, int anchor)
    {
        if (tool.Origin < 0) return false;
        var from = request.FindPhrase("from"u8, anchor);
        if (from < 0 && Wide(s, anchor) == -1) from = request.FindPhrase("from"u8);
        if (from < 0) return false;
        var clause = request.WordClause[from];
        var to = request.FindPhrase("to"u8, clause, from + 2);
        if (to < 0) return false;
        var end = to + 1;
        while (end < request.Words && request.WordClause[end] == clause && !RouteStop(request.Word(end))) end++;
        if (end == to + 1) return false;
        var changed = SetRoute(s, json, request, call, tool.Origin, from + 1, to - 1);
        changed |= SetRoute(s, json, request, call, tool.Destination, to + 1, end - 1);
        return changed;
    }

    static bool RouteStop(ReadOnlySpan<byte> w) =>
        w.SequenceEqual("at"u8) || w.SequenceEqual("by"u8) || w.SequenceEqual("for"u8) || w.SequenceEqual("in"u8) || w.SequenceEqual("on"u8) ||
        w.SequenceEqual("tomorrow"u8) || w.SequenceEqual("tonight"u8) || w.SequenceEqual("today"u8) || w.SequenceEqual("now"u8) || w.SequenceEqual("please"u8) ||
        w.SequenceEqual("asap"u8) || w.SequenceEqual("around"u8);

    static bool SetRoute(RepairScratch s, ReadOnlySpan<byte> json, RepairText request, int call, int param, int firstWord, int lastWord)
    {
        if (lastWord < firstWord) return false;
        var verbatim = request.SourceSpan(firstWord, lastWord);
        var arg = FindArg(s, call, param);
        if (arg >= 0)
        {
            if (s.Args[arg].Kind != ValueKind.String) return false;
            var currentLength = Normalized(s, StringContent(s, json, s.Args[arg]));
            var candidateLength = Normalized(s, verbatim, currentLength + 1);
            if (currentLength > 0 && Contains(s.Words.AsSpan(currentLength + 1, candidateLength), s.Words.AsSpan(0, currentLength))) return false;
            return SetString(s, ref s.Args[arg], verbatim);
        }
        var slot = AddArg(s, call, param);
        return slot >= 0 && SetString(s, ref s.Args[slot], verbatim);
    }

    static bool RepairBy(RepairTool tool, RepairScratch s, ReadOnlySpan<byte> json, RepairText request, int call, int anchor)
    {
        if (tool.SingleNumber < 0) return false;
        var by = request.FindPhrase("by"u8, anchor);
        if (by < 0 && Wide(s, anchor) == -1) by = request.FindPhrase("by"u8);
        if (by < 0 || by + 1 >= request.Words || !RepairText.TryParseDecimal(request.Word(by + 1), out var amount)) return false;
        var next = by + 2 < request.Words ? request.Word(by + 2) : default;
        if (next.SequenceEqual("pm"u8) || next.SequenceEqual("am"u8) || next.SequenceEqual("oclock"u8)) return false;
        var param = tool.Params[tool.SingleNumber];
        var arg = FindArg(s, call, tool.SingleNumber);
        if (arg >= 0 && s.Args[arg].Kind == ValueKind.Number && Utf8Parser.TryParse(Value(s, json, s.Args[arg]), out double current, out _) && Math.Abs(current - amount) < 1e-9) return false;
        if (arg < 0) arg = AddArg(s, call, tool.SingleNumber);
        if (arg < 0) return false;
        if (param.Kind == ParamKind.Integer && amount != Math.Floor(amount)) return false;
        return SetNumber(s, ref s.Args[arg], amount);
    }

    static bool FillMissing(RepairTool tool, RepairScratch s, ReadOnlySpan<byte> json, RepairText request, int call, int anchor)
    {
        var changed = false;
        for (var p = 0; p < tool.Params.Length; p++)
        {
            var param = tool.Params[p];
            if (!param.Required || FindArg(s, call, p) >= 0) continue;
            if (param.Options.Length > 0)
            {
                var named = NamedOption(param, request, anchor);
                if (named < 0) continue;
                var slot = AddArg(s, call, p);
                if (slot >= 0) { SetJson(s, ref s.Args[slot], param.Options[named], param.Options[named][0] == (byte)'"' ? ValueKind.String : ValueKind.Number); changed = true; }
            }
            else if (param.DefaultJson != null)
            {
                var slot = AddArg(s, call, p);
                if (slot >= 0) { SetJson(s, ref s.Args[slot], param.DefaultJson, ValueKind.Number); s.Args[slot].FromDefault = true; changed = true; }
            }
        }
        return changed;
    }

    static int FindArg(RepairScratch s, int call, int param)
    {
        if (param < 0) return -1;
        for (var a = 0; a < s.ArgCount; a++)
            if (s.Args[a].Call == call && s.Args[a].Param == param && !s.Args[a].Removed) return a;
        return -1;
    }

    static int AddArg(RepairScratch s, int call, int param)
    {
        if (s.ArgCount >= s.Args.Length) return -1;
        s.Args[s.ArgCount] = new ArgSlot { Call = call, Param = param, KeyStart = -1, Kind = ValueKind.Null, ValueStart = 0, ValueLength = 0, FromArena = true };
        return s.ArgCount++;
    }

    static void SetLiteral(RepairScratch s, ref ArgSlot arg, ValueKind kind) =>
        SetJson(s, ref arg, kind == ValueKind.True ? "true"u8 : "false"u8, kind);

    static void SetJson(RepairScratch s, ref ArgSlot arg, ReadOnlySpan<byte> value, ValueKind kind)
    {
        if (s.ArenaLength + value.Length > s.Arena.Length) return;
        value.CopyTo(s.Arena.AsSpan(s.ArenaLength));
        arg.ValueStart = s.ArenaLength;
        arg.ValueLength = value.Length;
        arg.Kind = kind;
        arg.FromArena = true;
        s.ArenaLength += value.Length;
    }

    static bool SetString(RepairScratch s, ref ArgSlot arg, ReadOnlySpan<byte> raw)
    {
        var start = s.ArenaLength;
        var at = start;
        var arena = s.Arena;
        if (at + raw.Length * 6 + 2 > arena.Length) return false;
        arena[at++] = (byte)'"';
        foreach (var c in raw)
        {
            if (c == (byte)'"' || c == (byte)'\\') { arena[at++] = (byte)'\\'; arena[at++] = c; }
            else if (c == (byte)'\n') { arena[at++] = (byte)'\\'; arena[at++] = (byte)'n'; }
            else if (c == (byte)'\r') { arena[at++] = (byte)'\\'; arena[at++] = (byte)'r'; }
            else if (c == (byte)'\t') { arena[at++] = (byte)'\\'; arena[at++] = (byte)'t'; }
            else if (c < 0x20) { arena[at++] = (byte)' '; }
            else arena[at++] = c;
        }
        arena[at++] = (byte)'"';
        arg.ValueStart = start;
        arg.ValueLength = at - start;
        arg.Kind = ValueKind.String;
        arg.FromArena = true;
        s.ArenaLength = at;
        return true;
    }

    static bool SetNumber(RepairScratch s, ref ArgSlot arg, double value)
    {
        if (s.ArenaLength + 32 > s.Arena.Length) return false;
        var destination = s.Arena.AsSpan(s.ArenaLength, 32);
        int written;
        var ok = value == Math.Floor(value) && Math.Abs(value) < 1e15
            ? Utf8Formatter.TryFormat((long)value, destination, out written)
            : Utf8Formatter.TryFormat(value, destination, out written);
        if (!ok) return false;
        arg.ValueStart = s.ArenaLength;
        arg.ValueLength = written;
        arg.Kind = ValueKind.Number;
        arg.FromArena = true;
        s.ArenaLength += written;
        return true;
    }

    static bool InRange(RepairTool tool, RepairScratch s, ReadOnlySpan<byte> json, int call)
    {
        for (var a = 0; a < s.ArgCount; a++)
        {
            ref var arg = ref s.Args[a];
            if (arg.Call != call || arg.Removed || arg.Param < 0 || arg.Kind != ValueKind.Number) continue;
            var param = tool.Params[arg.Param];
            if (!Utf8Parser.TryParse(Value(s, json, arg), out double value, out _)) continue;
            if (param.MinimumExclusive ? value <= param.Minimum : value < param.Minimum) return false;
            if (param.MaximumExclusive ? value >= param.Maximum : value > param.Maximum) return false;
        }
        return true;
    }

    static int Write(RepairPlan plan, RepairScratch s, ReadOnlySpan<byte> json, int callCount, out int kept, ref RepairOutcome outcome)
    {
        var spans = s.Spans;
        var o = s.Output;
        var at = 0;
        kept = 0;
        if (!Put(o, ref at, "["u8)) { return -1; }
        for (var c = 0; c < callCount; c++)
        {
            ref var call = ref s.Calls[c];
            if (call.Removed) continue;
            var tool = plan.Tools[call.Tool];
            var callStart = at;
            if (kept > 0 && !Put(o, ref at, ","u8)) return -1;
            if (!Put(o, ref at, "{\"name\":\""u8) || !Put(o, ref at, tool.EscapedName) || !Put(o, ref at, "\",\"arguments\":"u8)) return -1;
            var argsStart = at;
            if (!Put(o, ref at, "{"u8)) return -1;
            var first = true;
            for (var a = 0; a < s.ArgCount; a++)
            {
                ref var arg = ref s.Args[a];
                if (arg.Call != c || arg.Removed) continue;
                if (!first && !Put(o, ref at, ","u8)) return -1;
                first = false;
                if (!Put(o, ref at, "\""u8)) return -1;
                if (!Put(o, ref at, arg.KeyStart >= 0 ? json.Slice(arg.KeyStart, arg.KeyLength) : tool.Params[arg.Param].Key)) return -1;
                if (!Put(o, ref at, "\":"u8) || !Put(o, ref at, Value(s, json, arg))) return -1;
            }
            if (!Put(o, ref at, "}"u8)) return -1;
            var argsLength = at - argsStart;
            if (!Put(o, ref at, "}"u8)) return -1;
            var duplicate = false;
            for (var k = 0; k < kept; k++)
                if (spans[k].Tool == call.Tool && new ReadOnlySpan<byte>(o, spans[k].ArgumentsStart, spans[k].ArgumentsLength).SequenceEqual(new ReadOnlySpan<byte>(o, argsStart, argsLength)))
                {
                    duplicate = true;
                    break;
                }
            if (duplicate)
            {
                at = callStart;
                call.Removed = true;
                outcome |= RepairOutcome.Duplicate;
                continue;
            }
            spans[kept++] = new CallSpan(call.Tool, argsStart, argsLength);
            s.Calls[kept - 1].Tool = call.Tool;
        }
        if (!Put(o, ref at, "]"u8)) return -1;
        return at;
    }

    static bool Put(byte[] output, ref int at, ReadOnlySpan<byte> bytes)
    {
        if (at + bytes.Length > output.Length) return false;
        bytes.CopyTo(output.AsSpan(at));
        at += bytes.Length;
        return true;
    }

    static bool Ungrounded(RepairPlan plan, RepairScratch s, RepairText history, RepairText request, DateTime? today, int kept)
    {
        var spans = s.Spans;
        var found = false;
        var output = new ReadOnlySpan<byte>(s.Output);
        var anyYear = false;
        for (var n = 0; n < history.NumberCount; n++) anyYear |= IsYear(history.Numbers[n]);
        var relative = Relative(request);
        for (var k = 0; k < kept; k++)
        {
            var tool = plan.Tools[spans[k].Tool];
            var at = spans[k].ArgumentsStart + 1;
            var end = spans[k].ArgumentsStart + spans[k].ArgumentsLength - 1;
            while (at < end)
            {
                var keyStart = at + 1;
                if (!SkipString(output, ref at, end + 1)) break;
                var key = output.Slice(keyStart, at - 1 - keyStart);
                at++;
                var valueStart = at;
                if (!SkipValue(output, ref at, end, out var kind)) break;
                var value = output.Slice(valueStart, at - valueStart);
                var p = tool.FindParam(key);
                var param = p >= 0 ? tool.Params[p] : null;
                var flagged = false;
                if (kind == ValueKind.Number && param is { Options.Length: 0 } && Utf8Parser.TryParse(value, out double number, out _) &&
                    !history.HasNumber(number) && !(param.DefaultJson != null && value.SequenceEqual(param.DefaultJson)))
                    flagged = true;
                else if (kind == ValueKind.String && param is { Date: true } && anyYear && value.Length >= 6 && RepairText.TryParseDecimal(value.Slice(1, 4), out var year) &&
                         !history.HasNumber(year) && !(relative && today.HasValue && today.Value.Year == (int)year))
                    flagged = true;
                else if (kind == ValueKind.String && param is { FreeText: true } && value.Length > 2 &&
                         !(param.DefaultJson != null && value.SequenceEqual(param.DefaultJson)) && !WordsGrounded(s.Value, value[1..^1], history))
                    flagged = true;
                else if (kind == ValueKind.String && param is { NameOptions: true } && !NamedOption(param, value, history))
                    flagged = true;
                if (flagged)
                {
                    found = true;
                    AppendUngrounded(s, tool.EscapedName, key);
                }
                if (at < end && output[at] == (byte)',') at++;
            }
        }
        return found;
    }

    // 4.x grounding: free text must come from the conversation, every word of it ("05 minutes" for
    // "negative 5 minutes", or a resolved date for "yesterday", is not); semantics from the Apache-2.0
    // cactus-compute llms.txt ("Arguments contain only values evidenced in the input").
    static bool WordsGrounded(RepairText scratch, ReadOnlySpan<byte> text, RepairText history)
    {
        scratch.Load(text);
        for (var w = 0; w < scratch.Words; w++)
            if (!history.HasWordOrJoined(scratch.Word(w))) return false;
        return true;
    }

    // A closed set of names ("Maya", "Dr. Patel") is grounded only when the conversation names the chosen one.
    static bool NamedOption(RepairParam param, ReadOnlySpan<byte> value, RepairText history)
    {
        for (var o = 0; o < param.Options.Length; o++)
            if (value.SequenceEqual(param.Options[o])) return history.FindPhrase(param.OptionWords[o]) >= 0;
        return true;
    }

    static bool IsYear(double value) => value >= 1900 && value <= 2200 && value == Math.Floor(value);

    static bool Relative(RepairText request)
    {
        for (var w = 0; w < request.Words; w++)
        {
            var word = request.Word(w);
            if (word.SequenceEqual("today"u8) || word.SequenceEqual("tonight"u8) || word.SequenceEqual("tomorrow"u8) || word.SequenceEqual("yesterday"u8) ||
                word.SequenceEqual("next"u8) || word.SequenceEqual("this"u8) || word.SequenceEqual("coming"u8) || word.SequenceEqual("now"u8) || Weekday(word) >= 0)
                return true;
        }
        return false;
    }

    static void AppendUngrounded(RepairScratch s, ReadOnlySpan<byte> tool, ReadOnlySpan<byte> key)
    {
        var needed = tool.Length + key.Length + 2;
        if (s.UngroundedLength + needed > s.Ungrounded.Length) return;
        tool.CopyTo(s.Ungrounded.AsSpan(s.UngroundedLength));
        s.UngroundedLength += tool.Length;
        s.Ungrounded[s.UngroundedLength++] = (byte)'.';
        key.CopyTo(s.Ungrounded.AsSpan(s.UngroundedLength));
        s.UngroundedLength += key.Length;
        s.Ungrounded[s.UngroundedLength++] = (byte)'\n';
    }
}
