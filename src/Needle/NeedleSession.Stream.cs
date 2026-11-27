using System;

namespace Needle;

// The pieces a streamed turn needs from the session: start a turn, run it one step at a time, read how far it has
// got, and restore and repair the calls that have closed so far without touching the final answer's buffers.
public sealed unsafe partial class NeedleSession
{
    byte[]? provisionalInput;
    byte[]? provisionalOutput;
    byte[]? provisionalRaw;
    CallSpan[]? provisionalSpans;
    RepairScratch? provisionalScratch;

    /// <summary>True while a turn has started and not ended.</summary>
    internal bool TurnRunning => turn.Phase is not (TurnPhase.Idle or TurnPhase.Done);

    /// <summary>True once the current turn has ended; its reply is <see cref="TurnResult"/>.</summary>
    internal bool TurnEnded => turn.Phase == TurnPhase.Done;

    internal Reply TurnResult => turn.Result;

    /// <summary>Prompt tokens fed plus tokens decoded so far in this turn: the work a slice budget counts.</summary>
    internal int TurnWork => turn.PromptFed + turn.Progress.DecodeTokens;

    internal int TurnReasoningLength => turn.Progress.ReasoningLength;

    internal int TurnResponseLength => turn.Progress.ResponseLength;

    /// <summary>The constrained call text generated so far in this turn (tool and key aliases, not yet restored).</summary>
    internal ReadOnlySpan<byte> TurnCallText => turn.CallsReady ? new ReadOnlySpan<byte>(callText, 0, callTextLength) : default;

    internal byte[] CallTextBuffer => callText;

    internal ToolSet ToolSet => allTools;

    internal ThreadPolicy Threading => options.ResolvedThreading;

    /// <summary>The tool behind a generated (aliased) tool name.</summary>
    internal bool TryFindToolAlias(ReadOnlySpan<byte> alias, out int tool) => allTools.Aliases.TryFindTool(alias, out tool);

    /// <summary>Starts a turn to be run with <see cref="StepStreamed"/>. False when it ended before the model ran.</summary>
    internal bool BeginStreamed(ReadOnlySpan<byte> input, bool toolResult) => BeginTurn(input, toolResult, typed: false);

    /// <summary>Runs one step of the turn started with <see cref="BeginStreamed"/>. True when the turn has ended.</summary>
    internal bool StepStreamed(Span<byte> calls, Span<byte> reasoning, Span<byte> response) => StepTurn(calls, reasoning, response);

    /// <summary>Ends an unfinished turn as cancelled; like any cancelled turn, this forgets the conversation.</summary>
    internal void AbandonTurn()
    {
        if (TurnRunning) End(Abandon(ReplyStatus.Cancelled, FailureReason.None, turn.Progress));
    }

    internal void EnsureStreamBuffers()
    {
        if (provisionalInput != null) return;
        provisionalInput = new byte[callText.Length + 1];
        provisionalOutput = new byte[restored.Length];
        provisionalRaw = new byte[restored.Length];
        provisionalSpans = new CallSpan[callSpans.Length];
        if (repairPlan != null) provisionalScratch = new RepairScratch(callSpans.Length, Math.Max(64, callSpans.Length * 4), restored.Length + repairPlan.LongestValue * 8);
    }

    /// <summary>
    /// Restores and repairs the calls whose text ends at <paramref name="closeEnd"/> and reports the last of them as it
    /// stands with only the calls so far known. The final answer repairs the whole list again, so this is provisional:
    /// a later call can still change or drop it. Uses buffers of its own and never touches the final answer's.
    /// </summary>
    internal bool TryProvisionalCall(int closeEnd, out int tool, out ReadOnlySpan<byte> arguments, out bool withheld, out bool ungrounded)
    {
        EnsureStreamBuffers();
        tool = -1;
        arguments = default;
        withheld = false;
        ungrounded = false;
        if (!turn.CallsReady || closeEnd <= 0 || closeEnd > callTextLength) return false;
        var input = provisionalInput!;
        var output = provisionalOutput!;
        var spans = provisionalSpans!;
        new ReadOnlySpan<byte>(callText, 0, closeEnd).CopyTo(input);
        input[closeEnd] = (byte)']';
        if (!CallRestorer.TryRestore(allTools.Aliases, new ReadOnlySpan<byte>(input, 0, closeEnd + 1), output, keyScratch, spans, out var required, out var count)) return false;
        if (count == 0 || required > output.Length) return false;
        var last = count - 1;
        var raw = spans[last];
        tool = raw.Tool;
        new ReadOnlySpan<byte>(output, raw.ArgumentsStart, raw.ArgumentsLength).CopyTo(provisionalRaw);
        arguments = new ReadOnlySpan<byte>(provisionalRaw, 0, raw.ArgumentsLength);
        if (repairPlan == null) return true;
        var outcome = ArgumentRepair.Apply(repairPlan, provisionalScratch!, repairRequest!, repairHistory!, repairToday, output, required, spans, ref count, out var repaired);
        ungrounded = (outcome & RepairOutcome.Ungrounded) != 0;
        var rewritten = (outcome & (RepairOutcome.Repaired | RepairOutcome.Negated | RepairOutcome.Invalid | RepairOutcome.Duplicate)) != 0 && repaired <= output.Length;
        if (!rewritten) return true;
        if (provisionalScratch!.Calls[last].Removed)
        {
            withheld = true;
            return true;
        }
        if (count == 0) return true;
        var kept = spans[count - 1];
        tool = kept.Tool;
        arguments = new ReadOnlySpan<byte>(output, kept.ArgumentsStart, kept.ArgumentsLength);
        return true;
    }
}
