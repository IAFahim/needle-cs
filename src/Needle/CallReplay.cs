using System;
using Needle.Tools;

namespace Needle;

// Typed-path argument repair. The typed path decodes straight into user structs, so it also records the
// constrained bytes (the same aliased call text the JSON path keeps). After the call list closes, that text is
// restored to original names and repaired exactly as on the JSON path. When repair rewrote nothing, the commands
// the sink already produced stand. When it did, the repaired calls are replayed through a fresh grammar pass into
// a copy of the sink taken before the turn, translating original names back to the aliases the grammar expects.
internal static unsafe class CallReplay
{
    internal const RepairOutcome Rewrites = RepairOutcome.Repaired | RepairOutcome.Negated | RepairOutcome.Invalid | RepairOutcome.Duplicate;

    static ReadOnlySpan<byte> NameHead => "{\"name\":\""u8;
    static ReadOnlySpan<byte> ArgumentsHead => "\",\"arguments\":"u8;

    public static RepairOutcome Repair<TSink, TCommand>(GrammarProgram* program, AliasTable aliases, RepairPlan plan, RepairScratch scratch,
        RepairText request, RepairText history, DateTime? today, ReadOnlySpan<byte> callText, byte[] restored, Span<byte> keyScratch,
        Span<CallSpan> spans, in TSink before, ref TSink sink, Span<TCommand> commands, ref int stored, out int callCount, out int restoredLength,
        out FailureReason failure)
        where TSink : struct, ICallSink<TCommand>
        where TCommand : struct
    {
        failure = FailureReason.None;
        restoredLength = 0;
        if (!CallRestorer.TryRestore(aliases, callText, restored, keyScratch, spans, out var required, out callCount))
        {
            failure = FailureReason.MalformedCalls;
            return RepairOutcome.None;
        }
        if (required > restored.Length)
        {
            failure = FailureReason.CallTextTooLong;
            return RepairOutcome.None;
        }
        var outcome = ArgumentRepair.Apply(plan, scratch, request, history, today, restored, required, spans, ref callCount, out var repaired);
        restoredLength = repaired <= restored.Length ? repaired : required;
        if ((outcome & Rewrites) == 0) return outcome;
        // Apply only reports a rewrite once the repaired calls, spans and count are in place.
        sink = before;
        if (!Replay(program, aliases, new ReadOnlySpan<byte>(restored, 0, restoredLength), spans[..callCount], ref sink, commands, ref stored))
            failure = FailureReason.MalformedCalls;
        return outcome;
    }

    public static ReplyStatus Status(int callCount, float confidence, float refuseBelow, ulong route, RepairOutcome outcome)
    {
        // 4.x: a call below the floor is withheld unless a trigger routed it; a grounding gate withholds it either way.
        var refused = confidence < refuseBelow && route == 0 || (outcome & RepairOutcome.Ungrounded) != 0;
        return callCount == 0 ? ReplyStatus.Declined : refused ? ReplyStatus.Refused : ReplyStatus.Called;
    }

    // Drives restored calls (original names, compact JSON, as CallRestorer and ArgumentRepair write them) through
    // the grammar into the sink. Tool names come from the span's tool index; property keys are matched against the
    // aliases of the object the grammar is in, so a name reused at two depths finds the right alias at each.
    public static bool Replay<TSink, TCommand>(GrammarProgram* program, AliasTable aliases, ReadOnlySpan<byte> json, ReadOnlySpan<CallSpan> spans,
        ref TSink sink, Span<TCommand> commands, ref int stored)
        where TSink : struct, ICallSink<TCommand>
        where TCommand : struct
    {
        stored = 0;
        var state = Grammar.Start(program);
        var s = &state;
        if (!Feed(program, s, (byte)'[', ref sink, commands, ref stored)) return false;
        for (var k = 0; k < spans.Length; k++)
        {
            var span = spans[k];
            if (k > 0 && !Feed(program, s, (byte)',', ref sink, commands, ref stored)) return false;
            if (!Feed(program, s, NameHead, ref sink, commands, ref stored)) return false;
            if (!Feed(program, s, aliases.ToolAlias(span.Tool), ref sink, commands, ref stored)) return false;
            if (!Feed(program, s, ArgumentsHead, ref sink, commands, ref stored)) return false;
            if (span.ArgumentsStart < 0 || span.ArgumentsLength < 0 || span.ArgumentsStart + span.ArgumentsLength > json.Length) return false;
            if (!FeedArguments(program, s, aliases, span.Tool, json.Slice(span.ArgumentsStart, span.ArgumentsLength), ref sink, commands, ref stored)) return false;
            if (!Feed(program, s, (byte)'}', ref sink, commands, ref stored)) return false;
        }
        if (!Feed(program, s, (byte)']', ref sink, commands, ref stored)) return false;
        return Grammar.Complete(state);
    }

    static bool FeedArguments<TSink, TCommand>(GrammarProgram* program, GrammarState* s, AliasTable aliases, int tool, ReadOnlySpan<byte> arguments,
        ref TSink sink, Span<TCommand> commands, ref int stored)
        where TSink : struct, ICallSink<TCommand>
        where TCommand : struct
    {
        var at = 0;
        while (at < arguments.Length)
        {
            var c = arguments[at];
            if (c == (byte)'"' && Grammar.AwaitsKey(program, s, out var first, out var count, out var used))
            {
                var end = at + 1;
                while (end < arguments.Length && arguments[end] != (byte)'"') end += arguments[end] == (byte)'\\' ? 2 : 1;
                if (end >= arguments.Length) return false;
                var key = arguments.Slice(at + 1, end - at - 1);
                var name = key;
                for (var p = 0; p < count; p++)
                {
                    if ((used >> p & 1) != 0) continue;
                    var alias = Grammar.PropertyName(program, first + p);
                    if (aliases.TryFindProperty(tool, alias, out var original) ? original.SequenceEqual(key) : alias.SequenceEqual(key))
                    {
                        name = alias;
                        break;
                    }
                }
                if (!Feed(program, s, (byte)'"', ref sink, commands, ref stored) || !Feed(program, s, name, ref sink, commands, ref stored) ||
                    !Feed(program, s, (byte)'"', ref sink, commands, ref stored)) return false;
                at = end + 1;
                continue;
            }
            if (!Feed(program, s, c, ref sink, commands, ref stored)) return false;
            at++;
        }
        return true;
    }

    static bool Feed<TSink, TCommand>(GrammarProgram* program, GrammarState* s, ReadOnlySpan<byte> bytes, ref TSink sink, Span<TCommand> commands, ref int stored)
        where TSink : struct, ICallSink<TCommand>
        where TCommand : struct
    {
        for (var i = 0; i < bytes.Length; i++)
            if (!Feed(program, s, bytes[i], ref sink, commands, ref stored)) return false;
        return true;
    }

    static bool Feed<TSink, TCommand>(GrammarProgram* program, GrammarState* s, byte value, ref TSink sink, Span<TCommand> commands, ref int stored)
        where TSink : struct, ICallSink<TCommand>
        where TCommand : struct
    {
        if (!Grammar.Advance(program, s, value, ref sink)) return false;
        if (sink.TryTakeCall(out var call) && stored < commands.Length) commands[stored++] = call;
        return true;
    }
}
