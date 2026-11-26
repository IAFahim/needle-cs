using NUnit.Framework;
using UnityEngine;

namespace Needle.Unity.Tests
{
    // The designer assets without a model: the tool list becomes the JSON the engine reads, the
    // checks catch what a designer gets wrong, and the Agent asset maps to the engine's options.
    public sealed class AgentAssetTests
    {
        [Test]
        public void ToolListBecomesJsonTheEngineAccepts()
        {
            var tools = ScriptableObject.CreateInstance<NeedleToolSet>();
            var move = new NeedleToolDefinition { Name = "move_to", Description = "Walk to a \"place\".\nQuickly." };
            move.Parameters.Add(new NeedleToolParameter { Name = "place", Description = "Where to go." });
            move.Parameters.Add(new NeedleToolParameter { Name = "speed", Type = NeedleParameterType.Choice, Options = new[] { "walk", "run" }, Required = false });
            tools.Tools.Add(move);
            var wait = new NeedleToolDefinition { Name = "wait", Description = "Do nothing for a while." };
            wait.Parameters.Add(new NeedleToolParameter { Name = "seconds", Type = NeedleParameterType.Number });
            tools.Tools.Add(wait);

            Assert.IsEmpty(tools.Validate());
            string json = tools.ToJson();
            StringAssert.Contains("\"enum\":[\"walk\",\"run\"]", json);
            StringAssert.Contains("\"required\":[\"place\"]", json);
            StringAssert.Contains("Walk to a \\\"place\\\".\\nQuickly.", json);

            var parsed = new ToolSet(json);
            Assert.AreEqual(2, parsed.Tools.Count);

            tools.Json = new TextAsset(json);
            CollectionAssert.AreEqual(new[] { "move_to", "wait" }, tools.ToolNames());
            Object.DestroyImmediate(tools);
        }

        [Test]
        public void ValidationNamesWhatADesignerGotWrong()
        {
            var tools = ScriptableObject.CreateInstance<NeedleToolSet>();
            Assert.IsNotEmpty(tools.Validate(), "an empty set");
            tools.Tools.Add(new NeedleToolDefinition { Name = "Move To", Description = "" });
            var problems = tools.Validate();
            Assert.IsTrue(problems.Exists(p => p.Contains("snake_case")), string.Join("; ", problems));
            Assert.IsTrue(problems.Exists(p => p.Contains("no description")), string.Join("; ", problems));
            Object.DestroyImmediate(tools);
        }

        [Test]
        public void AgentAssetMapsToEngineOptions()
        {
            var agent = ScriptableObject.CreateInstance<NeedleAgentAsset>();
            AgentOptions options = agent.ToOptions();
            Assert.AreEqual(0.3f, options.RefuseBelow, 1e-6f);
            Assert.IsTrue(options.Stateless);
            Assert.AreEqual(2, options.Threading.Value.Threads);
            agent.Remember = true;
            agent.Think = false;
            agent.RefuseBelow = 0f;
            options = agent.ToOptions();
            Assert.IsFalse(options.Stateless);
            Assert.AreEqual(AgentOptions.NoReasoning, options.ReasoningBudget);
            Assert.Less(options.RefuseBelow, 0f, "0 in the Inspector means never refuse");
            Object.DestroyImmediate(agent);
        }
    }
}
