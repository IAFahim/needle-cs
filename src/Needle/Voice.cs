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
        vocabulary = EnumKeywords(toolsJson);
    }

    // Startup-only convenience: accepts the UTF-8 schema bytes (for example the bytes
    // of a W10 generated tool set's Schema) and decodes them once here.
    public VoiceSession(Model needleModel, Model whistleModel, ReadOnlySpan<byte> toolsJson, NeedleOptions? options = null)
        : this(needleModel, whistleModel, System.Text.Encoding.UTF8.GetString(toolsJson), options)
    {
    }

    public NeedleSession Needle => needle;

    public SpeechSpan TranscribeSpan(ReadOnlySpan<float> pcm, Span<byte> utf8, Span<WordSlice> words, string? language = null) =>
        whistle.TranscribeUtf8(pcm, utf8, words, language, vocabulary);

    // Zero-allocation voice-to-call: span transcription with enum-literal keyword bias,
    // then one span request through the session. Silence yields an EmptyInput reply.
    public (Reply Reply, SpeechSpan Speech) CompleteSpan(ReadOnlySpan<float> pcm, Span<byte> calls, Span<byte> reasoning, Span<byte> response, Span<byte> transcriptUtf8, Span<WordSlice> transcriptWords, string? language = null)
    {
        var speech = TranscribeSpan(pcm, transcriptUtf8, transcriptWords, language);
        var reply = needle.Complete(transcriptUtf8.Slice(0, speech.Utf8Length), calls, reasoning, response);
        return (reply, speech);
    }


    internal static System.Collections.Generic.List<string> EnumKeywords(string toolsJson) => EnumValues(new ToolSet(toolsJson));

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
