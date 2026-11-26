using UnityEditor;
using UnityEngine;

namespace Needle.EditorTools
{
    [CustomEditor(typeof(NeedleToolSet))]
    sealed class NeedleToolSetEditor : UnityEditor.Editor
    {
        bool showJson;

        public override void OnInspectorGUI()
        {
            var set = (NeedleToolSet)target;
            EditorGUILayout.HelpBox("List what the agent can do. A clear one-sentence description per tool and per parameter is what makes the model pick the right one.", MessageType.None);
            DrawDefaultInspector();

            var problems = set.Validate();
            if (problems.Count > 0)
                EditorGUILayout.HelpBox(string.Join("\n", problems), MessageType.Warning);

            EditorGUILayout.Space();
            showJson = EditorGUILayout.Foldout(showJson, "Tools as the model sees them (JSON)", true);
            if (showJson)
            {
                string json = set.ToJson();
                EditorGUILayout.SelectableLabel(json, EditorStyles.textArea, GUILayout.MinHeight(80));
                if (GUILayout.Button("Copy JSON")) EditorGUIUtility.systemCopyBuffer = json;
            }
        }
    }
}
