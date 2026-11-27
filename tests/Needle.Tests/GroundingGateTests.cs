using System;
using System.Text;
using Xunit;

namespace Needle.Tests;

// 4.x grounding gate and negation cases from the Apache-2.0 cactus-compute 4.2.0 acceptance suites
// (tests/eval/environments): values the request never states are flagged, so the session withholds the call.
public sealed class GroundingGateTests
{
    const string Tools = """
[{"name":"set_timer","parameters":{"type":"object","properties":{"time_human":{"type":"string","minLength":1}},"required":["time_human"]}},
 {"name":"create_reminder","parameters":{"type":"object","properties":{"message":{"type":"string"},"date_time_human":{"type":"string","description":"The user's date or time phrase copied word for word."}},"required":["message","date_time_human"]}},
 {"name":"reply_to_notification","parameters":{"type":"object","properties":{"notification_match":{"type":"string","enum":["Maya","Leo","Dr. Patel","Cactus Team"]},"text":{"type":"string"}},"required":["notification_match","text"]}},
 {"name":"dismiss_notification","parameters":{"type":"object","properties":{"notification_match":{"type":"string","enum":["Maya","Leo","Dr. Patel","Cactus Team"]}},"required":["notification_match"]}},
 {"name":"set_cooking_timer","parameters":{"type":"object","properties":{"label":{"type":"string","description":"Label for the timer, copied from the user."},"minutes":{"type":"integer","minimum":1}},"required":["label","minutes"]}},
 {"name":"log_contact","parameters":{"type":"object","properties":{"email":{"type":"string","format":"email"}},"required":["email"]}}]
""";

    static (string Json, RepairOutcome Outcome, string Ungrounded) Run(string query, string generated, string system = "")
    {
        var set = new ToolSet(Tools);
        var plan = new RepairPlan(set, Tools);
        var scratch = new RepairScratch(16, 64, 4096);
        var request = new RepairText(1024);
        var history = new RepairText(8192);
        history.Load(Encoding.UTF8.GetBytes(system));
        var q = Encoding.UTF8.GetBytes(query);
        request.Load(q);
        history.Append(q);
        var buffer = new byte[4096];
        var spans = new CallSpan[16];
        Assert.True(CallRestorer.TryRestore(set.Aliases, Encoding.UTF8.GetBytes(generated), buffer, new byte[64], spans, out var length, out var count));
        var outcome = ArgumentRepair.Apply(plan, scratch, request, history, new DateTime(2026, 11, 27), buffer, length, spans, ref count, out var required);
        return (Encoding.UTF8.GetString(buffer, 0, required), outcome, Encoding.UTF8.GetString(scratch.Ungrounded, 0, scratch.UngroundedLength).Trim());
    }

    static string Call(string name, string args) => $$"""[{"name":"{{name}}","arguments":{{args}}}]""";

    [Fact]
    public void FreeTextTheRequestNeverSaysIsUngrounded()
    {
        var (_, outcome, ungrounded) = Run("set a timer for negative 5 minutes", Call("set_timer", """{"time_human":"05 minutes"}"""));
        Assert.True(outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.Equal("set_timer.time_human", ungrounded);
    }

    [Fact]
    public void FreeTextCopiedFromTheRequestIsGrounded()
    {
        Assert.False(Run("start a timer for 25 minutes", Call("set_timer", """{"time_human":"25 minutes"}""")).Outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.False(Run("set a timer that goes off at 9:15 pm", Call("set_timer", """{"time_human":"9:15pm"}""")).Outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.False(Run("set an alarm for seven in the morning", Call("set_timer", """{"time_human":"7:00 AM"}""")).Outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.False(Run("wake me at seven pm", Call("set_timer", """{"time_human":"19:00"}""")).Outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.False(Run("timer until noon", Call("set_timer", """{"time_human":"12:00 pm"}""")).Outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.True(Run("set a timer for 5 minutes", Call("set_timer", """{"time_human":"8 minutes"}""")).Outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.False(Run("log jane.doe@example.com", Call("log_contact", """{"email":"jane.doe@example.com"}""")).Outcome.HasFlag(RepairOutcome.Ungrounded));
    }

    [Fact]
    public void HumanPhraseIsNotResolvedToADateAndAnInventedDateIsUngrounded()
    {
        var generated = Call("create_reminder", """{"message":"submit the timesheet","date_time_human":"2026-11-28 16:00"}""");
        var (json, outcome, ungrounded) = Run("remind me yesterday at 4pm to submit the timesheet", generated, "date: 2026-11-27 Fri 09:30");
        Assert.Contains("2026-11-28 16:00", json);
        Assert.True(outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.Equal("create_reminder.date_time_human", ungrounded);
        var kept = Run("remind me tomorrow at 7am to call mom", Call("create_reminder", """{"message":"call mom","date_time_human":"tomorrow at 7am"}"""), "date: 2026-11-27 Fri 09:30");
        Assert.Contains("tomorrow at 7am", kept.Json);
        Assert.False(kept.Outcome.HasFlag(RepairOutcome.Ungrounded));
    }

    [Fact]
    public void NameTheRequestNeverSaysIsUngrounded()
    {
        var (_, outcome, ungrounded) = Run("reply to Priya saying happy birthday", Call("reply_to_notification", """{"notification_match":"Maya","text":"happy birthday"}"""));
        Assert.True(outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.Equal("reply_to_notification.notification_match", ungrounded);
        Assert.False(Run("reply to Dr. Patel saying see you then", Call("reply_to_notification", """{"notification_match":"Dr. Patel","text":"see you then"}""")).Outcome.HasFlag(RepairOutcome.Ungrounded));
    }

    [Fact]
    public void NeverBeforeATitledNameNegatesTheCall()
    {
        var (json, outcome, _) = Run("never dismiss notifications from Dr. Patel", Call("dismiss_notification", """{"notification_match":"Dr. Patel"}"""));
        Assert.Equal("[]", json);
        Assert.True(outcome.HasFlag(RepairOutcome.Negated));
    }

    [Fact]
    public void TheSystemTextDoesNotStateANumber()
    {
        const string system = "date: 2026-11-27 Fri 02:11; Map each explicit action to exactly one declared call.";
        var (_, outcome, ungrounded) = Run("set a cooking timer for the rice", Call("set_cooking_timer", """{"label":"rice","minutes":1}"""), system);
        Assert.True(outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.Equal("set_cooking_timer.minutes", ungrounded);
        Assert.True(Run("set a cooking timer for the rice", Call("set_cooking_timer", """{"label":"rice","minutes":11}"""), system).Outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.False(Run("set a rice timer for one minute", Call("set_cooking_timer", """{"label":"rice","minutes":1}"""), system).Outcome.HasFlag(RepairOutcome.Ungrounded));
    }

    [Fact]
    public void ALabelThatOnlyRestatesTheToolIsUngrounded()
    {
        var (_, outcome, ungrounded) = Run("start a timer for 25 minutes", Call("set_cooking_timer", """{"label":"timer","minutes":25}"""));
        Assert.True(outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.Equal("set_cooking_timer.label", ungrounded);
        Assert.False(Run("start a pasta timer for 9 minutes", Call("set_cooking_timer", """{"label":"pasta","minutes":9}""")).Outcome.HasFlag(RepairOutcome.Ungrounded));
    }
}
