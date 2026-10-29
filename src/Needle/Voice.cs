using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Needle;

public sealed class VoiceResult
{
    public Transcription Speech { get; init; } = new();
    public CallResult Call { get; init; } = new();

    public string ToJson()
    {
        var call = Call.ToJson();
        var b = new StringBuilder(call, 0, call.Length - 1, call.Length + 256);
        b.Append(",\"audio_text\":");
        JsonValue.WriteString(b, Speech.Text);
        b.Append(",\"audio_language\":");
        if (Speech.Language == null) b.Append("null"); else JsonValue.WriteString(b, Speech.Language);
        b.Append(",\"audio_ttft_ms\":").Append(Speech.TimeToFirstTokenMs.ToString("0.##", CultureInfo.InvariantCulture));
        b.Append(",\"audio_decode_tps\":").Append(Speech.DecodeTps.ToString("0.#", CultureInfo.InvariantCulture)).Append('}');
        return b.ToString();
    }
}

public sealed class VoiceSession : IDisposable
{
    readonly Whistle whistle;
    readonly NeedleSession needle;
    readonly List<string> vocabulary;

    public VoiceSession(Model needleModel, Model whistleModel, string toolsJson, NeedleOptions? options = null)
    {
        whistle = new Whistle(whistleModel, options?.ResolvedThreading ?? ThreadPolicy.Default);
        needle = new NeedleSession(needleModel, toolsJson, options);
        vocabulary = EnumValues(new ToolSet(toolsJson));
    }

    public NeedleSession Needle => needle;

    public VoiceResult Complete(ReadOnlySpan<float> pcm, string? language = null)
    {
        var speech = whistle.Transcribe(pcm, language, vocabulary);
        var call = speech.Text.Length == 0
            ? new CallResult { Reasoning = "no speech detected", Confidence = 0f }
            : needle.Complete(speech.Text);
        return new VoiceResult { Speech = speech, Call = call };
    }

    static List<string> EnumValues(ToolSet tools)
    {
        var found = new List<string>();
        void Walk(JsonValue v)
        {
            if (v.Kind == JsonKind.Object)
                foreach (var m in v.Members)
                {
                    if (m.Key == "enum" && m.Value.Kind == JsonKind.Array)
                        foreach (var option in m.Value.Items)
                            if (option.Kind == JsonKind.String && !found.Contains(option.Text)) found.Add(option.Text);
                    Walk(m.Value);
                }
            else if (v.Kind == JsonKind.Array) foreach (var i in v.Items) Walk(i);
        }
        foreach (var tool in tools.Tools) Walk(tool);
        return found;
    }

    public void Pause()
    {
        whistle.Pause();
        needle.Pause();
    }

    public void Resume()
    {
        whistle.Resume();
        needle.Resume();
    }

    public void Dispose()
    {
        whistle.Dispose();
        needle.Dispose();
    }
}
