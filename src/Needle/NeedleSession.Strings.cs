using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Needle;

public sealed unsafe partial class NeedleSession
{
    const int CallsHeadroomBytes = 256;

    sealed class StringBridge
    {
        public readonly byte[] Input;
        public readonly byte[] Calls;
        public readonly byte[] Reasoning;
        public readonly byte[] Response;

        public StringBridge(int inputBytes, int callBytes, int reasoningBytes, int responseBytes)
        {
            Input = new byte[inputBytes];
            Calls = new byte[callBytes];
            Reasoning = new byte[reasoningBytes];
            Response = new byte[responseBytes];
        }
    }

    StringBridge? bridge;

    public CallResult Complete(string query) => Turn(query, false);

    public CallResult CompleteWithToolResult(string resultJson) => Turn(resultJson, true);

    CallResult Turn(string text, bool toolResult)
    {
        var buffers = bridge ??= new StringBridge(maxInputBytes, callText.Length + CallsHeadroomBytes,
            Math.Max(0, options.MaxReasoningTokens) * MaxPieceBytes, Math.Max(0, options.MaxCallTokens) * MaxPieceBytes);
        text ??= "";
        var byteCount = Encoding.UTF8.GetByteCount(text);
        Reply reply;
        if (byteCount > buffers.Input.Length) reply = Failed(FailureReason.InputTooLong, new TurnProgress { Started = Stopwatch.GetTimestamp() });
        else
        {
            Encoding.UTF8.GetBytes(text.AsSpan(), buffers.Input);
            reply = Run(new ReadOnlySpan<byte>(buffers.Input, 0, byteCount), toolResult, buffers.Calls, buffers.Reasoning, buffers.Response);
        }
        return ToCallResult(reply, buffers);
    }

    CallResult ToCallResult(in Reply reply, StringBridge buffers)
    {
        var ticksPerSecond = (double)Stopwatch.Frequency;
        var reasoning = Decode(buffers.Reasoning, reply.ReasoningLength);
        if (reply.Status == ReplyStatus.Cancelled) return new CallResult { Success = false, Error = "cancelled", Reasoning = reasoning };
        if (reply.Status == ReplyStatus.Failed && reply.Failure != FailureReason.CallsBufferTooSmall)
            return new CallResult { Success = false, Error = Describe(reply.Failure), Reasoning = reasoning };

        var timing = new CallResult
        {
            PrefillTokens = reply.PromptTokens,
            DecodeTokens = reply.DecodeTokens,
            TimeToFirstTokenMs = reply.FirstTokenTicks * 1000.0 / ticksPerSecond,
            PrefillTps = reply.PromptTokens / Math.Max(1e-9, reply.PrefillTicks / ticksPerSecond),
            DecodeTps = reply.DecodeTokens / Math.Max(1e-9, (reply.ElapsedTicks - reply.PrefillTicks) / ticksPerSecond),
            TotalMs = reply.ElapsedTicks * 1000.0 / ticksPerSecond,
        };
        if (reply.Status == ReplyStatus.Responded)
            return new CallResult
            {
                Reasoning = reasoning,
                Response = Decode(buffers.Response, reply.ResponseLength),
                Confidence = reply.Confidence,
                PrefillTokens = timing.PrefillTokens,
                DecodeTokens = timing.DecodeTokens,
                TimeToFirstTokenMs = timing.TimeToFirstTokenMs,
                PrefillTps = timing.PrefillTps,
                DecodeTps = timing.DecodeTps,
                TotalMs = timing.TotalMs,
            };

        var json = buffers.Calls;
        var refused = reply.Status == ReplyStatus.Refused;
        if (reply.Failure == FailureReason.CallsBufferTooSmall)
        {
            json = new byte[reply.Utf8Length];
            CallRestorer.TryRestore(allTools.Aliases, new ReadOnlySpan<byte>(callText, 0, callTextLength), json, keyScratch, callSpans, out _, out _);
            refused = reply.Confidence < options.RefuseBelow;
        }
        var calls = new FunctionCall[reply.CallCount];
        for (var i = 0; i < calls.Length; i++)
        {
            var span = callSpans[i];
            calls[i] = new FunctionCall(allTools.Aliases.OriginalTool(span.Tool), Encoding.UTF8.GetString(json, span.ArgumentsStart, span.ArgumentsLength));
        }
        return new CallResult
        {
            FunctionCalls = refused ? Array.Empty<FunctionCall>() : calls,
            SuppressedCalls = refused ? calls : Array.Empty<FunctionCall>(),
            Reasoning = reasoning,
            Confidence = reply.Confidence,
            HeadConfidence = reply.HeadConfidence,
            DecodeConfidence = reply.DecodeConfidence,
            PrefillTokens = timing.PrefillTokens,
            DecodeTokens = timing.DecodeTokens,
            TimeToFirstTokenMs = timing.TimeToFirstTokenMs,
            PrefillTps = timing.PrefillTps,
            DecodeTps = timing.DecodeTps,
            TotalMs = timing.TotalMs,
        };
    }

    static string Decode(byte[] buffer, int length) => Encoding.UTF8.GetString(buffer, 0, length).Trim();

    static string Describe(FailureReason reason) => reason switch
    {
        FailureReason.EmptyInput => "the query is empty",
        FailureReason.InputTooLong => "the input exceeds the maximum size",
        FailureReason.ContextExhausted => "context capacity exceeded",
        FailureReason.CallTokenBudgetExceeded => "call exceeded the token budget",
        FailureReason.CallTextTooLong => "the generated call exceeds the call text buffer",
        FailureReason.GrammarDeadEnd => "the grammar left no allowed token",
        FailureReason.GrammarRejectedToken => "grammar rejected an allowed token",
        FailureReason.CallsBufferTooSmall => "the calls buffer is too small",
        FailureReason.MalformedCalls => "the generated calls are not valid JSON",
        FailureReason.ToolContextFailed => "the tool context could not be built",
        _ => "the turn failed",
    };
}
