using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Needle
{
    /// <summary>
    /// A Needle model file in StreamingAssets. Every agent that points at the same file shares one
    /// open copy of it; the file is mapped, not read, so opening it is cheap.
    /// </summary>
    [CreateAssetMenu(menuName = "Needle/Model", fileName = "Needle Model", order = 100)]
    public sealed class NeedleModelAsset : ScriptableObject
    {
        [Tooltip("The model's file name in Assets/StreamingAssets, for example needle3.cact.")]
        public string FileName = "needle3.cact";

        [Tooltip("Touch every page of the model when it opens, so the first answer does not wait on the disk. Costs a moment at load.")]
        public bool Prefault = true;

        [Tooltip("Optional: the file's size in bytes. On Android, where the file is copied out of the APK once, a copy of another size is refused.")]
        public long ExpectedBytes = -1;

        /// <summary>The full path the model opens from, or null while it still has to be copied out of an Android APK.</summary>
        public string DirectPath
        {
            get
            {
                if (string.IsNullOrEmpty(FileName)) return null;
                if (Path.IsPathRooted(FileName)) return FileName;
                string streaming = ModelLoader.StreamingPath(FileName);
                if (!ModelLoader.IsApkUrl(streaming)) return streaming;
                string cached = ModelLoader.CachedPath(FileName);
                return File.Exists(cached) ? cached : null;
            }
        }

        /// <summary>
        /// A coroutine that makes the model openable from a path: nothing to do on desktop, and on
        /// Android a one-time copy out of the APK. Calls <paramref name="onReady"/> with the path,
        /// or <paramref name="onError"/> with a message.
        /// </summary>
        public IEnumerator Prepare(Action<string> onReady, Action<string> onError)
        {
            string path = DirectPath;
            if (path != null)
            {
                onReady(path);
                yield break;
            }
            string error = null;
            yield return ModelLoader.OpenAsync(FileName, model => model.Dispose(), e => error = e, true, false, ExpectedBytes);
            if (error != null) onError(error);
            else onReady(ModelLoader.CachedPath(FileName));
        }

        /// <summary>Opens the model at <paramref name="path"/>, or adds a user to the copy already open.</summary>
        public static NeedleModel Acquire(string path, bool prefault)
        {
            lock (open)
            {
                Shared shared;
                if (open.TryGetValue(path, out shared))
                {
                    shared.Users++;
                    return shared.Model;
                }
                var model = NeedleModel.Open(path, prefault);
                open[path] = new Shared { Model = model, Users = 1 };
                return model;
            }
        }

        /// <summary>Gives back a model from <see cref="Acquire"/>; the last user closes it.</summary>
        public static void Release(string path)
        {
            lock (open)
            {
                Shared shared;
                if (path == null || !open.TryGetValue(path, out shared)) return;
                if (--shared.Users > 0) return;
                open.Remove(path);
                shared.Model.Dispose();
            }
        }

        sealed class Shared
        {
            public NeedleModel Model;
            public int Users;
        }

        static readonly Dictionary<string, Shared> open = new Dictionary<string, Shared>(StringComparer.Ordinal);
    }
}
