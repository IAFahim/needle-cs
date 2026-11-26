using System;
using System.Diagnostics;

namespace Needle;

public enum ReplyStatus : byte { Called, Responded, Refused, Failed, Cancelled, Declined }

public enum FailureReason : byte
{
    None,
    EmptyInput,
    InputTooLong,
    ContextExhausted,
    CallTokenBudgetExceeded,
    CallTextTooLong,
    GrammarDeadEnd,
    GrammarRejectedToken,
    CallsBufferTooSmall,
    MalformedCalls,
    ToolContextFailed,
    FrameBudgetExceeded,
}

[Flags]
public enum ReplyFlags : byte
{
    None = 0,
    ReasoningTruncated = 1,
    ResponseTruncated = 2,
    Repaired = 4,
    Negated = 8,
    Invalid = 16,
    Ungrounded = 32,
}

public readonly record struct Reply(
    ReplyStatus Status,
    FailureReason Failure,
    ReplyFlags Flags,
    float Confidence,
    float HeadConfidence,
    float DecodeConfidence,
    int CallCount,
    int Utf8Length,
    int ReasoningLength,
    int ResponseLength,
    int PromptTokens,
    int DecodeTokens,
    long PrefillTicks,
    long FirstTokenTicks,
    long ElapsedTicks)
{
    public static long TicksPerSecond => Stopwatch.Frequency;

    public bool HasCalls => Status is ReplyStatus.Called or ReplyStatus.Refused;
}
