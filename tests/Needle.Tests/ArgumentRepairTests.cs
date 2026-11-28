using System;
using System.Text;
using Xunit;

namespace Needle.Tests;

public sealed class ArgumentRepairTests
{
    const string Home = """
[{"name":"turn_on_light","description":"Turn on a light.","parameters":{"type":"object","properties":{"room":{"type":"string"}},"required":["room"]}},
 {"name":"turn_off_light","description":"Turn off a light.","parameters":{"type":"object","properties":{"room":{"type":"string"}},"required":["room"]}},
 {"name":"set_brightness","description":"Dim or brighten a light.","parameters":{"type":"object","properties":{"room":{"type":"string"},"level":{"type":"integer","minimum":0,"maximum":100}},"required":["room","level"]}},
 {"name":"set_thermostat","description":"Set the heating mode.","parameters":{"type":"object","properties":{"mode":{"type":"string","enum":["heat","cool","auto","off"]},"temperature":{"type":"number"}},"required":["mode"]}},
 {"name":"lock_door","description":"Lock or unlock a door.","parameters":{"type":"object","properties":{"door":{"type":"string"},"locked":{"type":"boolean"}},"required":["door","locked"]}},
 {"name":"send_message","description":"Send a text.","parameters":{"type":"object","properties":{"recipient":{"type":"string"},"message":{"type":"string"}},"required":["recipient","message"]}},
 {"name":"create_contact","description":"Add a contact.","parameters":{"type":"object","properties":{"first_name":{"type":"string"},"last_name":{"type":"string"},"phone":{"type":"string"}},"required":["first_name"]}},
 {"name":"book_ride","description":"Book a ride.","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"}},"required":["destination"]}},
 {"name":"create_event","description":"Add an event.","parameters":{"type":"object","properties":{"title":{"type":"string"},"date":{"type":"string"}},"required":["title","date"]}},
 {"name":"adjust_volume","description":"Change the volume.","parameters":{"type":"object","properties":{"delta":{"type":"integer"}},"required":["delta"]}}]
""";

    sealed class Rig
    {
        readonly ToolSet set = new(Home);
        readonly RepairPlan plan;
        readonly RepairScratch scratch = new(16, 64, 4096);
        readonly RepairText request = new(1024);
        readonly RepairText history = new(8192);

        public Rig(string system = "")
        {
            plan = new RepairPlan(set, Home);
            history.Load(Encoding.UTF8.GetBytes(system));
        }

        public (string Json, RepairOutcome Outcome, string Ungrounded) Run(string query, string generated, DateTime? today = null)
        {
            var q = Encoding.UTF8.GetBytes(query);
            request.Load(q);
            history.Append(q);
            var buffer = new byte[4096];
            var spans = new CallSpan[16];
            Assert.True(CallRestorer.TryRestore(set.Aliases, Encoding.UTF8.GetBytes(generated), buffer, new byte[64], spans, out var length, out var count));
            var outcome = ArgumentRepair.Apply(plan, scratch, request, history, today, buffer, length, spans, ref count, out var required);
            var ungrounded = Encoding.UTF8.GetString(scratch.Ungrounded, 0, scratch.UngroundedLength).Replace('\n', ' ').Trim();
            return (Encoding.UTF8.GetString(buffer, 0, required), outcome, ungrounded);
        }
    }

    static string Call(string name, string args) => $$"""[{"name":"{{name}}","arguments":{{args}}}]""";

    [Fact]
    public void CleanCallsStayByteIdentical()
    {
        var generated = Call("set_brightness", """{"room":"bedroom","level":40}""");
        var (json, outcome, _) = new Rig().Run("dim the bedroom lights to 40 percent", generated);
        Assert.Equal(generated, json);
        Assert.Equal(RepairOutcome.None, outcome);
    }

    [Fact]
    public void PairedToolFollowsTheVerb()
    {
        var (json, outcome, _) = new Rig().Run("turn off the kitchen light", Call("turn_on_light", """{"room":"kitchen"}"""));
        Assert.Equal(Call("turn_off_light", """{"room":"kitchen"}"""), json);
        Assert.True(outcome.HasFlag(RepairOutcome.Repaired));
    }

    [Fact]
    public void NegatedRequestDropsTheCall()
    {
        var (json, outcome, _) = new Rig().Run("don't turn on the kitchen light", Call("turn_on_light", """{"room":"kitchen"}"""));
        Assert.Equal("[]", json);
        Assert.True(outcome.HasFlag(RepairOutcome.Negated));
    }

    [Fact]
    public void NegationOnlyDropsItsOwnClause()
    {
        var generated = """[{"name":"turn_on_light","arguments":{"room":"kitchen"}},{"name":"turn_off_light","arguments":{"room":"hallway"}}]""";
        var (json, _, _) = new Rig().Run("turn on the kitchen light but don't turn off the hallway light", generated);
        Assert.Equal(Call("turn_on_light", """{"room":"kitchen"}"""), json);
    }

    [Fact]
    public void DontForgetIsNotNegation()
    {
        var generated = Call("send_message", """{"recipient":"Sam","message":"dinner at 7"}""");
        var (json, _, _) = new Rig().Run("text Sam: don't forget dinner at 7", generated);
        Assert.Equal(generated, json);
    }

    [Fact]
    public void OutOfRangeNumberDropsTheCall()
    {
        var (json, outcome, _) = new Rig().Run("dim the bedroom lights to 150 percent", Call("set_brightness", """{"room":"bedroom","level":150}"""));
        Assert.Equal("[]", json);
        Assert.True(outcome.HasFlag(RepairOutcome.Invalid));
    }

    [Fact]
    public void BooleanFollowsTheVerb()
    {
        var (json, _, _) = new Rig().Run("unlock the front door", Call("lock_door", """{"door":"front","locked":true}"""));
        Assert.Equal(Call("lock_door", """{"door":"front","locked":false}"""), json);
    }

    [Fact]
    public void EnumMovesToTheNamedOption()
    {
        var (json, _, _) = new Rig().Run("switch the thermostat to cool", Call("set_thermostat", """{"mode":"heat"}"""));
        Assert.Equal(Call("set_thermostat", """{"mode":"cool"}"""), json);
    }

    [Fact]
    public void UnstatedOptionalNumberIsDropped()
    {
        var (json, _, _) = new Rig().Run("switch the thermostat to cool", Call("set_thermostat", """{"mode":"cool","temperature":21}"""));
        Assert.Equal(Call("set_thermostat", """{"mode":"cool"}"""), json);
    }

    [Fact]
    public void FullNameIsSplit()
    {
        var (json, _, _) = new Rig().Run("add Ada Lovelace to my contacts", Call("create_contact", """{"first_name":"Ada Lovelace"}"""));
        Assert.Equal(Call("create_contact", """{"first_name":"Ada","last_name":"Lovelace"}"""), json);
    }

    [Fact]
    public void TruncatedPhoneIsCompleted()
    {
        var (json, _, _) = new Rig().Run("save Bo with number 555-123-4567", Call("create_contact", """{"first_name":"Bo","phone":"555-123"}"""));
        Assert.Equal(Call("create_contact", """{"first_name":"Bo","phone":"555-123-4567"}"""), json);
    }

    [Fact]
    public void QuotedTitleIsVerbatim()
    {
        var (json, _, _) = new Rig("date: 2026-11-26 Thu 09:00").Run("add \"Q4 Planning – Kickoff\" to my calendar today",
            Call("create_event", """{"title":"q4 planning kickoff","date":"2026-11-26"}"""), new DateTime(2026, 11, 26));
        Assert.Equal(Call("create_event", """{"title":"Q4 Planning – Kickoff","date":"2026-11-26"}"""), json);
    }

    [Fact]
    public void RelativeDateResolvesAgainstToday()
    {
        var (json, _, _) = new Rig("date: 2026-11-26 Thu 09:00").Run("schedule dentist for tomorrow",
            Call("create_event", """{"title":"dentist","date":"2025-01-01"}"""), new DateTime(2026, 11, 26));
        Assert.Equal(Call("create_event", """{"title":"dentist","date":"2026-11-27"}"""), json);
    }

    [Fact]
    public void WeekdayResolvesForward()
    {
        Assert.True(ArgumentRepair.ResolveDate(Text("lunch next monday"), new DateTime(2026, 11, 26), -1, out var date));
        Assert.Equal(new DateTime(2026, 11, 30), date);
        Assert.True(ArgumentRepair.ResolveDate(Text("party on March 4th"), new DateTime(2026, 11, 26), -1, out date));
        Assert.Equal(new DateTime(2027, 3, 4), date);
        Assert.True(ArgumentRepair.ResolveDate(Text("in 3 days"), new DateTime(2026, 11, 26), -1, out date));
        Assert.Equal(new DateTime(2026, 11, 29), date);
    }

    [Fact]
    public void RouteFillsOriginAndDestination()
    {
        var (json, _, _) = new Rig().Run("get me a ride from Union Station to the airport", Call("book_ride", """{"destination":"airport"}"""));
        Assert.Equal(Call("book_ride", """{"destination":"airport","origin":"Union Station"}"""), json);
    }

    [Fact]
    public void ByAmountFillsTheSingleNumber()
    {
        var (json, _, _) = new Rig().Run("turn the volume up by 5", Call("adjust_volume", """{"delta":10}"""));
        Assert.Equal(Call("adjust_volume", """{"delta":5}"""), json);
    }

    [Fact]
    public void DuplicateCallsCollapse()
    {
        var generated = """[{"name":"turn_on_light","arguments":{"room":"kitchen"}},{"name":"turn_on_light","arguments":{"room":"kitchen"}}]""";
        var (json, outcome, _) = new Rig().Run("turn on the kitchen light", generated);
        Assert.Equal(Call("turn_on_light", """{"room":"kitchen"}"""), json);
        Assert.True(outcome.HasFlag(RepairOutcome.Duplicate));
    }

    [Fact]
    public void UnstatedRequiredNumberIsReportedUngrounded()
    {
        var (_, outcome, ungrounded) = new Rig().Run("dim the bedroom lights", Call("set_brightness", """{"room":"bedroom","level":30}"""));
        Assert.True(outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.Equal("set_brightness.level", ungrounded);
    }

    [Fact]
    public void ValuesStayInTheirOwnClause()
    {
        var generated = """[{"name":"adjust_volume","arguments":{"delta":30}},{"name":"set_brightness","arguments":{"room":"bedroom","level":10}}]""";
        var (json, _, _) = new Rig().Run("change the volume by 30 and dim the bedroom lights to 10", generated);
        Assert.Equal(generated, json);
    }

    [Fact]
    public void NegationInsideTheMessageIsContent()
    {
        var generated = Call("send_message", """{"recipient":"Sam","message":"I won't make it"}""");
        var (json, outcome, _) = new Rig().Run("text Sam that I won't make it", generated);
        Assert.Equal(generated, json);
        Assert.False(outcome.HasFlag(RepairOutcome.Negated));
    }

    [Fact]
    public void OrAfterANegationNegatesBoth()
    {
        var generated = """[{"name":"turn_on_light","arguments":{"room":"kitchen"}},{"name":"turn_on_light","arguments":{"room":"hallway"}}]""";
        var (json, _, _) = new Rig().Run("don't turn on the kitchen light or the hallway light", generated);
        Assert.Equal("[]", json);
    }

    [Fact]
    public void FullHistoryKeepsTheNewestTurn()
    {
        var history = new RepairText(64);
        history.Load(Encoding.UTF8.GetBytes("an old line about the garden that fills the buffer"));
        history.Append(Encoding.UTF8.GetBytes("set the bedroom to 21"));
        Assert.True(history.HasNumber(21));
        Assert.True(history.FindPhrase("bedroom"u8) >= 0);
    }

    [Fact]
    public void PassedMonthAndDayMeansNextYear()
    {
        var request = new RepairText(256);
        request.Load(Encoding.UTF8.GetBytes("add dentist on January 5"));
        Assert.True(ArgumentRepair.ResolveDate(request, new DateTime(2026, 11, 26), -1, out var date));
        Assert.Equal(new DateTime(2027, 1, 5), date);
    }

    [Fact]
    public void ParamRolesNeedTheRightName()
    {
        const string tools = """
[{"name":"schedule","description":"Schedule a meeting.","parameters":{"type":"object","properties":{"start":{"type":"string","description":"Start time"},"end":{"type":"string","description":"End time"},"update":{"type":"string"},"dueDate":{"type":"string"},"number":{"type":"string","description":"Seat number"}},"required":["start"]}}]
""";
        var plan = new RepairPlan(new ToolSet(tools), tools);
        var tool = plan.Tools[0];
        Assert.Equal(-1, tool.Origin);
        Assert.False(tool.Params[tool.FindParam("update"u8)].Date);
        Assert.True(tool.Params[tool.FindParam("dueDate"u8)].Date);
        Assert.False(tool.Params[tool.FindParam("number"u8)].Phone);
    }

    [Fact]
    public void OutputThatDoesNotFitLeavesCallsUntouched()
    {
        var set = new ToolSet(Home);
        var plan = new RepairPlan(set, Home);
        var request = new RepairText(1024);
        var history = new RepairText(8192);
        var q = Encoding.UTF8.GetBytes("get me a ride from Union Station to the airport");
        request.Load(q);
        history.Load(q);
        var buffer = new byte[4096];
        var spans = new CallSpan[16];
        Assert.True(CallRestorer.TryRestore(set.Aliases, Encoding.UTF8.GetBytes(Call("book_ride", """{"destination":"airport"}""")), buffer, new byte[64], spans, out var length, out var count));
        var before = spans[0];
        var original = Encoding.UTF8.GetString(buffer, 0, length);
        var outcome = ArgumentRepair.Apply(plan, new RepairScratch(16, 64, 4096), request, history, null, buffer.AsSpan(0, length), length, spans, ref count, out var required);
        Assert.True(required > length);
        Assert.Equal(1, count);
        Assert.Equal(before, spans[0]);
        Assert.Equal(original, Encoding.UTF8.GetString(buffer, 0, length));
        Assert.False(outcome.HasFlag(RepairOutcome.Repaired));
    }

    [Fact]
    public void RepairDoesNotAllocateOnceWarm()
    {
        var rig = new Rig();
        rig.Run("turn off the kitchen light", Call("turn_on_light", """{"room":"kitchen"}"""));
        var set = new ToolSet(Home);
        var plan = new RepairPlan(set, Home);
        var scratch = new RepairScratch(16, 64, 4096);
        var request = new RepairText(1024);
        var history = new RepairText(8192);
        var q = Encoding.UTF8.GetBytes("turn off the kitchen light and unlock the front door");
        var generated = Encoding.UTF8.GetBytes("""[{"name":"turn_on_light","arguments":{"room":"kitchen"}},{"name":"lock_door","arguments":{"door":"front","locked":true}}]""");
        var buffer = new byte[4096];
        var spans = new CallSpan[16];
        var key = new byte[64];
        void Once()
        {
            request.Load(q);
            history.Load(q);
            CallRestorer.TryRestore(set.Aliases, generated, buffer, key, spans, out var length, out var count);
            ArgumentRepair.Apply(plan, scratch, request, history, new DateTime(2026, 11, 26), buffer, length, spans, ref count, out _);
        }
        Once();
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 100; i++) Once();
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    static RepairText Text(string text)
    {
        var t = new RepairText(1024);
        t.Load(Encoding.UTF8.GetBytes(text));
        return t;
    }
}
