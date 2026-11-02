// N64: build-time primed-prefix baker (Editor menu for `needle prime`).
// Bakes the post-prefill KV snapshot for a (model, tool set, system prompt,
// context) tuple and writes it to a .prefix file. Ship the file (for example
// in StreamingAssets) and load it with the NeedleSession(model, tools,
// prefixBytes, options) constructor or HostOptions.PrimedPrefixPath to skip
// the tool-prompt prefill at startup: session ready in milliseconds.
//
// The bake is date-free (AutoDate=false, plan.md section 10.5): a dated
// system prompt would hash differently the next day and the load would fail
// validation. The load validates model geometry, vocab, context, tool set
// and system prompt and throws on mismatch; rebake when any of those change.
// Retrieval stays off (full tool set is primed): loading hosts need
// RetrievalTopK at or above the tool count.
// C# 9 safe: block namespaces, no target-typed new, no u8 literals.
using System;
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEngine;
using Needle;

namespace Needle.Editor
{
    public static class NeedlePrefixBaker
    {
        [MenuItem("Needle/Bake Primed Prefix...")]
        public static void BakeFromDialog()
        {
            string modelPath = EditorUtility.OpenFilePanel("Needle model (.cact)", Application.streamingAssetsPath, "cact");
            if (string.IsNullOrEmpty(modelPath))
            {
                return;
            }
            string toolsPath = EditorUtility.OpenFilePanel("Tool set (.json)", Path.GetDirectoryName(modelPath), "json");
            if (string.IsNullOrEmpty(toolsPath))
            {
                return;
            }
            string outputPath = EditorUtility.SaveFilePanel("Save primed prefix", Path.GetDirectoryName(modelPath), Path.GetFileNameWithoutExtension(modelPath) + ".prefix", "prefix");
            if (string.IsNullOrEmpty(outputPath))
            {
                return;
            }
            BakePrimedPrefix(modelPath, File.ReadAllText(toolsPath), outputPath);
        }

        public static string BakePrimedPrefix(string modelPath, string toolsJson, string outputPath)
        {
            return BakePrimedPrefix(modelPath, toolsJson, outputPath, string.Empty, 2048);
        }

        public static string BakePrimedPrefix(string modelPath, string toolsJson, string outputPath, string system, int context)
        {
            if (string.IsNullOrEmpty(modelPath))
            {
                throw new ArgumentException("model path is required", "modelPath");
            }
            if (string.IsNullOrEmpty(toolsJson))
            {
                throw new ArgumentException("tool set JSON is required", "toolsJson");
            }
            if (string.IsNullOrEmpty(outputPath))
            {
                throw new ArgumentException("output path is required", "outputPath");
            }
            if (context < 1)
            {
                throw new ArgumentOutOfRangeException("context");
            }

            // Retrieval stays off for the bake: the full tool set is primed,
            // and a primed prefix cannot be combined with tool retrieval.
            NeedleOptions options = new NeedleOptions
            { // full tool-set prime; retrieval stays off
                Context = context,
                System = system,
                AutoDate = false,
                RetrievalTopK = int.MaxValue,
                Threading = ThreadPolicy.Game.WithThreads(Math.Clamp(Environment.ProcessorCount / 2, 1, 8)),
            };

            Stopwatch build = Stopwatch.StartNew();
            using (Model model = Model.Open(modelPath))
            {
                int tokens;
                byte[] bytes;
                using (NeedleSession session = new NeedleSession(model, toolsJson, options))
                {
                    build.Stop();
                    tokens = session.ContextUsed;
                    bytes = session.ExportPrimedPrefix();
                }
                File.WriteAllBytes(outputPath, bytes);

                // Validate the file: a fresh session must restore it and agree
                // on the primed length. Timed as the N64 ready measurement.
                Stopwatch restore = Stopwatch.StartNew();
                using (NeedleSession revived = new NeedleSession(model, toolsJson, bytes, options))
                {
                    restore.Stop();
                    if (revived.ContextUsed != tokens)
                    {
                        throw new InvalidOperationException("primed prefix mismatch: baked " + tokens + " tokens, restored " + revived.ContextUsed);
                    }
                }
                double buildMs = build.Elapsed.TotalMilliseconds;
                double restoreMs = restore.Elapsed.TotalMilliseconds;
                UnityEngine.Debug.Log("[Needle] primed prefix: " + tokens + " tokens in " + buildMs.ToString("F1") + " ms, wrote " + bytes.Length + " bytes -> " + outputPath);
                UnityEngine.Debug.Log("[Needle] primed prefix: restored " + tokens + " tokens in " + restoreMs.ToString("F1") + " ms (context " + context + ")");
                if (restoreMs > 50)
                {
                    UnityEngine.Debug.LogWarning("[Needle] primed prefix restore took " + restoreMs.ToString("F1") + " ms, above the 50 ms N64 budget on this machine.");
                }
            }
            AssetDatabase.Refresh();
            return outputPath;
        }
    }
}
