using System.IO;
using Needle;
using UnityEngine;

namespace Needle.Samples
{
    public class VoiceSample : MonoBehaviour
    {
        public string ClipFile = "test.wav";
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
            whistle = new Whistle(model, 1);
            clip = Wav.Read(clipPath);
        }
        void Update()
        {
            if (whistle == null) return;
            if (Input.GetKeyDown(KeyCode.V)) { talk.Hold(); head = 0; held = true; }
            if (held)
            {
                var rooms = new System.Span<float>(clip, head, System.Math.Min(1600, clip.Length - head));
                held = talk.Write(rooms) && head + 1600 < clip.Length;
                head += rooms.Length;
                if (!held) Release();
            }
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
