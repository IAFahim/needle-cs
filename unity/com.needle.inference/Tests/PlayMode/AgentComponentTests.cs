using System;
using System.Collections;
using System.Diagnostics;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Needle.Unity.PlayMode.Tests
{
    // The designer path: assets made in code the way a designer makes them in the Inspector, a
    // Needle Agent component on a GameObject, an order, and the tool's events firing with the
    // argument already converted. Ignored without the model, unless NEEDLE_REQUIRE_MODEL=1.
    public sealed class AgentComponentTests
    {
        GameObject owner;

        [TearDown]
        public void TearDown()
        {
            if (owner != null) UnityEngine.Object.Destroy(owner);
            owner = null;
        }

        [UnityTest]
        public IEnumerator AnOrderFiresTheToolsEventsWithItsArguments()
        {
            string model = PlayModeAssets.Require(PlayModeAssets.FromEnvOrRepo("NEEDLE_MODEL", "ref/hf/needle3.cact"), "Needle 3 model (needle3.cact)");

            var modelAsset = ScriptableObject.CreateInstance<NeedleModelAsset>();
            modelAsset.FileName = System.IO.Path.GetFullPath(model);
            modelAsset.Prefault = false;

            var tools = ScriptableObject.CreateInstance<NeedleToolSet>();
            var timer = new NeedleToolDefinition { Name = "set_timer", Description = "Start a countdown timer." };
            timer.Parameters.Add(new NeedleToolParameter { Name = "minutes", Type = NeedleParameterType.Integer, Description = "How long the timer runs, in minutes." });
            tools.Tools.Add(timer);
            var message = new NeedleToolDefinition { Name = "send_message", Description = "Send a text message to a contact." };
            message.Parameters.Add(new NeedleToolParameter { Name = "recipient", Description = "Who gets the message." });
            message.Parameters.Add(new NeedleToolParameter { Name = "message", Description = "What the message says." });
            tools.Tools.Add(message);
            Assert.IsEmpty(tools.Validate());

            var agentAsset = ScriptableObject.CreateInstance<NeedleAgentAsset>();
            agentAsset.Model = modelAsset;
            agentAsset.Tools = tools;
            agentAsset.Threads = PlayModeAssets.Threads;

            owner = new GameObject("Needle agent test");
            owner.SetActive(false);
            var agent = owner.AddComponent<NeedleAgentComponent>();
            agent.Agent = agentAsset;
            agent.LogAnswers = true;

            var binding = new NeedleToolBinding { Tool = "set_timer" };
            var minutes = new NeedleArgumentBinding { Parameter = "minutes", Kind = NeedleArgumentKind.WholeNumber };
            binding.Arguments.Add(minutes);
            agent.Tools.Add(binding);

            int called = 0, gotMinutes = -1, calls = 0;
            bool ready = false;
            string failed = null, unsure = null, toolName = null;
            binding.Called.AddListener(() => called++);
            minutes.WholeNumber.AddListener(value => gotMinutes = value);
            agent.OnReady.AddListener(() => ready = true);
            agent.OnCall.AddListener((tool, arguments) => { calls++; toolName = tool; });
            agent.OnFailed.AddListener(reason => failed = reason);
            agent.OnUnsure.AddListener(line => unsure = line);

            owner.SetActive(true);
            // Given before the agent has loaded: it is answered once the agent is ready.
            agent.Ask("set a timer for 12 minutes");

            var wall = Stopwatch.StartNew();
            while (calls == 0 && failed == null && unsure == null)
            {
                if (wall.Elapsed > TimeSpan.FromMinutes(3)) Assert.Fail("no answer within three minutes (ready: " + ready + ")");
                yield return null;
            }

            Assert.IsNull(failed, "the agent failed: " + failed);
            Assert.IsNull(unsure, "the agent was unsure: " + agent.LastResult);
            Assert.IsTrue(ready, "On Ready did not run before the answer");
            Assert.AreEqual("set_timer", toolName);
            Assert.AreEqual(1, called, "the set_timer binding's Called event");
            Assert.AreEqual(12, gotMinutes, "the minutes argument");
            Assert.AreEqual(AnswerKind.Calls, agent.LastResult.Kind);
        }
    }
}
