using System.Collections.Generic;
using UnityEngine;

namespace Needle.Samples
{
    /// <summary>
    /// A text box for giving a Needle Agent orders and a log of what it did. It only talks to the
    /// agent through public members, so it works with any Agent asset.
    /// </summary>
    [AddComponentMenu("Needle/Samples/Order Panel")]
    public sealed class OrderPanel : MonoBehaviour
    {
        public NeedleAgentComponent Agent;

        [TextArea(2, 6)]
        public string Hint = "Try: move to 120, 45 / give Lyra 3 potions / switch to wedge formation and follow the captain";

        readonly List<string> log = new List<string>();
        string order = string.Empty;
        Vector2 scroll;

        void OnEnable()
        {
            if (Agent == null) Agent = GetComponent<NeedleAgentComponent>();
            if (Agent == null) return;
            Agent.Answered += OnAnswered;
            Agent.OnUnsure.AddListener(Say);
            Agent.OnReply.AddListener(Say);
            Agent.OnFailed.AddListener(Fail);
        }

        void OnDisable()
        {
            if (Agent == null) return;
            Agent.Answered -= OnAnswered;
            Agent.OnUnsure.RemoveListener(Say);
            Agent.OnReply.RemoveListener(Say);
            Agent.OnFailed.RemoveListener(Fail);
        }

        void OnAnswered(NeedleAgentResult result)
        {
            if (result.Kind == AnswerKind.Calls)
                foreach (NeedleAgentCall call in result.Calls)
                    Add("→ " + call);
            Add("   (" + result.Milliseconds.ToString("0") + " ms, confidence " + result.Confidence.ToString("0.00") + ")");
        }

        void Say(string line) => Add("“" + line + "”");

        void Fail(string reason) => Add("failed: " + reason);

        void Add(string line)
        {
            log.Add(line);
            if (log.Count > 200) log.RemoveAt(0);
            scroll.y = float.MaxValue;
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, Mathf.Min(Screen.width - 20, 640), Screen.height - 20), GUI.skin.box);
            GUILayout.Label(Agent == null ? "No Needle Agent" : Agent.IsReady ? (Agent.IsBusy ? "Thinking…" : "Ready") : "Loading…");
            GUILayout.Label(Hint);
            GUILayout.BeginHorizontal();
            GUI.SetNextControlName("order");
            order = GUILayout.TextField(order);
            bool send = GUILayout.Button("Ask", GUILayout.Width(60))
                || (Event.current.type == EventType.KeyDown && Event.current.keyCode == KeyCode.Return && GUI.GetNameOfFocusedControl() == "order");
            GUILayout.EndHorizontal();
            if (send && Agent != null && !string.IsNullOrWhiteSpace(order))
            {
                Add("> " + order);
                Agent.Ask(order);
                order = string.Empty;
            }
            scroll = GUILayout.BeginScrollView(scroll);
            foreach (string line in log) GUILayout.Label(line);
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }
    }
}
