using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace Needle.EditorTools
{
    [CustomEditor(typeof(NeedleAgentComponent))]
    sealed class NeedleAgentComponentEditor : UnityEditor.Editor
    {
        string order = string.Empty;

        public override void OnInspectorGUI()
        {
            var agent = (NeedleAgentComponent)target;
            serializedObject.Update();
            DrawDefaultInspector();

            NeedleToolSet tools = agent.Agent != null ? agent.Agent.Tools : null;
            if (tools != null)
            {
                string[] names = tools.ToolNames();
                var missing = new List<string>();
                foreach (string name in names)
                    if (agent.Tools.FindIndex(b => b.Tool == name) < 0) missing.Add(name);
                var unknown = new List<string>();
                foreach (NeedleToolBinding binding in agent.Tools)
                    if (System.Array.IndexOf(names, binding.Tool) < 0) unknown.Add(string.IsNullOrEmpty(binding.Tool) ? "(empty)" : binding.Tool);

                if (unknown.Count > 0)
                    EditorGUILayout.HelpBox("These bindings name no tool in '" + tools.name + "': " + string.Join(", ", unknown), MessageType.Warning);
                if (missing.Count > 0)
                {
                    EditorGUILayout.HelpBox("Tools with nothing wired up yet: " + string.Join(", ", missing), MessageType.Info);
                    if (GUILayout.Button("Add a binding for every tool"))
                    {
                        Undo.RecordObject(agent, "Add Needle tool bindings");
                        foreach (string name in missing) agent.Tools.Add(NewBinding(tools, name));
                        EditorUtility.SetDirty(agent);
                    }
                }
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Try it", EditorStyles.boldLabel);
            using (new EditorGUI.DisabledScope(!Application.isPlaying))
            {
                EditorGUILayout.BeginHorizontal();
                order = EditorGUILayout.TextField(order);
                if (GUILayout.Button("Ask", GUILayout.Width(60)) && !string.IsNullOrWhiteSpace(order)) agent.Ask(order);
                EditorGUILayout.EndHorizontal();
            }
            if (!Application.isPlaying)
                EditorGUILayout.HelpBox("Enter Play mode to give the agent orders from here.", MessageType.None);
            else
            {
                string state = agent.IsReady ? (agent.IsBusy ? "Answering…" : "Ready") : "Loading…";
                EditorGUILayout.LabelField("State", state);
                NeedleAgentResult last = agent.LastResult;
                if (last != null)
                {
                    EditorGUILayout.LabelField("Last order", last.Order);
                    EditorGUILayout.LabelField("Answer", last.ToString(), EditorStyles.wordWrappedLabel);
                    EditorGUILayout.LabelField("Confidence", last.Confidence.ToString("0.00") + "   (" + last.Milliseconds.ToString("0") + " ms)");
                }
                Repaint();
            }
        }

        static NeedleToolBinding NewBinding(NeedleToolSet tools, string name)
        {
            var binding = new NeedleToolBinding { Tool = name };
            foreach (NeedleToolDefinition tool in tools.Tools)
            {
                if (tool.Name != name) continue;
                foreach (NeedleToolParameter parameter in tool.Parameters)
                    binding.Arguments.Add(new NeedleArgumentBinding { Parameter = parameter.Name, Kind = KindFor(parameter.Type) });
            }
            return binding;
        }

        static NeedleArgumentKind KindFor(NeedleParameterType type)
        {
            switch (type)
            {
                case NeedleParameterType.Integer: return NeedleArgumentKind.WholeNumber;
                case NeedleParameterType.Number: return NeedleArgumentKind.Number;
                case NeedleParameterType.Flag: return NeedleArgumentKind.Flag;
                default: return NeedleArgumentKind.Text;
            }
        }
    }
}
