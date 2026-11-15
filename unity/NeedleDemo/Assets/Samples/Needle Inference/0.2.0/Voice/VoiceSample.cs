using System.IO;
using Needle;
using UnityEngine;

namespace Needle.Samples
{
    public class VoiceSample : MonoBehaviour
    {
        public string ClipFile = "test.wav";
        [Tooltip("Whistle worker threads. 0 = automatic: half the logical cores, capped at 4 (4 threads where the box has headroom, 2 on a 4-core laptop). Set 1-4 to force a count.")]
        public int Threads = 0;
        Whistle whistle;
        Model model;
        readonly PushToTalk talk = new PushToTalk();
        readonly byte[] text = new byte[1024];
        readonly WordSlice[] words = new WordSlice[32];
        float[] clip;
        int head;
        bool held;
        void OnEnable()
        {
            var modelPath = Path.Combine(Application.streamingAssetsPath, "whistle.cact");
            var clipPath = Path.Combine(Application.streamingAssetsPath, ClipFile);
            if (!File.Exists(modelPath)) { Debug.LogError($"Whistle model missing: {modelPath}"); enabled = false; return; }
            if (!File.Exists(clipPath)) { Debug.LogError($"Voice clip missing: {clipPath}"); enabled = false; return; }
            model = Model.Open(modelPath);
            whistle = new Whistle(model, ThreadPolicy.Game.WithThreads(ResolveThreads()));
            clip = Wav.Read(clipPath);
        }
        int ResolveThreads() => Threads > 0 ? System.Math.Max(1, System.Math.Min(4, Threads)) : DefaultThreads();

        static int DefaultThreads()
        {
            int cores = System.Math.Max(1, System.Environment.ProcessorCount);
            return System.Math.Max(1, System.Math.Min(4, cores / 2));
        }
        public bool TriggerClip()
        {
            if (whistle == null || clip == null) return false;
            talk.Hold();
            head = 0;
            held = true;
            return true;
        }
        void Update()
        {
            if (whistle == null) return;
#if ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKeyDown(KeyCode.V)) TriggerClip();
#endif
            if (held)
            {
                var rooms = new System.Span<float>(clip, head, System.Math.Min(1600, clip.Length - head));
                held = talk.Write(rooms) && head + 1600 < clip.Length;
                head += rooms.Length;
                if (!held) Release();
            }
        }
        void OnGUI()
        {
            if (whistle == null) return;
            if (GUI.Button(new Rect(16f, 16f, 220f, 34f), "Run Voice Sample (" + ClipFile + ")"))
                TriggerClip();
        }
        void Release()
        {
            var speech = talk.Release(whistle, text, words);
            Debug.Log($"Heard {speech.Utf8Length} UTF-8 bytes; truncated={speech.Truncated}.");
        }
        void OnDisable()
        {
            whistle?.Dispose();
            whistle = null;
            model?.Dispose();
            model = null;
        }
    }
}
