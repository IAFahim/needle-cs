using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace Needle
{
    /// <summary>
    /// Model delivery per platform (N81). <see cref="Model.Open"/> mmaps a real
    /// file, which covers the Editor and every desktop/mobile player — except
    /// Android, where StreamingAssets live inside the APK and are not files.
    ///
    /// <para>Use <see cref="OpenAsync"/> everywhere: it mmaps directly where
    /// the OS exposes a file and downloads-then-opens inside an APK.</para>
    /// </summary>
    public static class ModelLoader
    {
        /// <summary>StreamingAssets path for a bundled model file.</summary>
        public static string StreamingPath(string fileName) =>
            Path.Combine(Application.streamingAssetsPath, fileName);

        /// <summary>
        /// True when <paramref name="path"/> addresses the inside of an APK
        /// (<c>jar:file://…!/assets</c>) rather than a real file. Pure string
        /// test, so it stays true in the Editor when pointed at an APK URL.
        /// </summary>
        public static bool IsApkUrl(string path) =>
            path.StartsWith("jar:file://", StringComparison.Ordinal) ||
            path.IndexOf("!/assets", StringComparison.Ordinal) >= 0;

        /// <summary>
        /// True when the file can be opened directly with <see cref="Model.Open"/>.
        /// False on Android (APK URL), or when the file is simply absent.
        /// </summary>
        public static bool CanOpenDirect(string fileName)
        {
            var path = StreamingPath(fileName);
            return !IsApkUrl(path) && File.Exists(path);
        }

        /// <summary>
        /// Persistent copy of a bundled model. Survives app restarts; delete it
        /// (or bump <paramref name="fileName"/>) when the bundled model changes.
        /// </summary>
        public static string CachedPath(string fileName) =>
            Path.Combine(Application.persistentDataPath, fileName);

        /// <summary>
        /// Open a model at an absolute file path. Desktop players, iOS and the
        /// Editor land here: <see cref="Model.Open"/> mmaps where the runtime
        /// supports it and reads where it does not (ns2.1 fallback).
        /// Throws on APK URLs — use <see cref="OpenAsync"/> there.
        /// </summary>
        public static Model OpenDirect(string absolutePath, bool prefault = true)
        {
            if (IsApkUrl(absolutePath))
                throw new IOException(
                    $"Model '{absolutePath}' is inside the APK and is not a file. " +
                    $"Load it with ModelLoader.OpenAsync instead.");
            return Model.Open(absolutePath, prefault);
        }

        /// <summary>
        /// Load a model bundled as <c>StreamingAssets/&lt;fileName&gt;</c> on any
        /// platform. Calls <paramref name="onReady"/> on the main thread with an
        /// open <see cref="Model"/> the caller owns; calls <paramref name="onError"/>
        /// (or logs) when the model cannot be delivered. Start this from a loading
        /// scene: first open faults pages and blocks for a large-model read.
        /// </summary>
        /// <param name="fileName">
        /// Model file under StreamingAssets, e.g. <c>"needle3.cact"</c>.
        /// </param>
        /// <param name="onReady">Receives the open model; dispose it when done.</param>
        /// <param name="onError">Receives a human-readable reason. Defaults to Debug.LogError.</param>
        /// <param name="cache">
        /// Android APK only: copy the download to <c>persistentDataPath</c> once,
        /// then <see cref="Model.Open"/> the copy so later launches mmap like
        /// desktop. When false, <see cref="Model.FromBytes"/> holds the model.
        /// </param>
        /// <param name="prefault">Fault model pages in on open (first-open latency now, fewer hitches later).</param>
        /// <param name="expectedBytes">
        /// Optional integrity check: a download or cache whose length differs is
        /// rejected and, for the cache, re-downloaded. -1 disables the check.
        /// </param>
        public static IEnumerator OpenAsync(
            string fileName,
            Action<Model> onReady,
            Action<string> onError = null,
            bool cache = true,
            bool prefault = true,
            long expectedBytes = -1)
        {
            if (string.IsNullOrEmpty(fileName))
            {
                Fail("ModelLoader needs a StreamingAssets file name, e.g. \"needle3.cact\".", onError);
                yield break;
            }
            if (onReady == null)
            {
                Fail("ModelLoader needs an onReady callback to receive the model.", onError);
                yield break;
            }

            // Direct-file platforms: Editor, Windows/macOS/Linux players, iOS.
            // This is also the whole Linux-player path the N81 acceptance needs.
            var streaming = StreamingPath(fileName);
            if (!IsApkUrl(streaming))
            {
                Model direct = null;
                string directError = null;
                try
                {
                    direct = Model.Open(streaming, prefault);
                }
                catch (Exception e)
                {
                    directError = $"Could not open model '{streaming}': {e.Message}";
                }
                if (directError != null) Fail(directError, onError);
                else onReady(direct);
                yield break;
            }

            // Android: StreamingAssets are inside the APK, so download them out.
            var cached = CachedPath(fileName);
            if (cache)
            {
                if (TryOpenCache(cached, expectedBytes, prefault, onReady, out var cacheError))
                    yield break;
                if (cacheError != null)
                {
                    Fail(cacheError, onError);
                    yield break;
                }
            }

            byte[] bytes;
            using (var request = UnityWebRequest.Get(streaming))
            {
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    Fail($"Could not download model '{streaming}': {request.error}", onError);
                    yield break;
                }
                bytes = request.downloadHandler.data;
            }

            if (bytes == null || bytes.Length == 0)
            {
                Fail($"Downloaded model '{streaming}' is empty.", onError);
                yield break;
            }
            if (expectedBytes >= 0 && bytes.Length != expectedBytes)
            {
                Fail($"Downloaded model '{streaming}' has {bytes.Length} bytes, expected {expectedBytes}; refusing a truncated copy.", onError);
                yield break;
            }

            if (!cache)
            {
                FromBytes(bytes, streaming, onReady, onError);
                yield break;
            }

            string writeError = WriteCacheAtomically(cached, bytes);
            bytes = null;
            if (writeError != null)
            {
                Fail(writeError, onError);
                yield break;
            }
            if (!TryOpenCache(cached, expectedBytes, prefault, onReady, out var openError))
                Fail(openError ?? $"Could not open cached model '{cached}'.", onError);
        }

        static bool TryOpenCache(
            string cached,
            long expectedBytes,
            bool prefault,
            Action<Model> onReady,
            out string error)
        {
            error = null;
            if (!File.Exists(cached)) return false;
            if (expectedBytes >= 0 && new FileInfo(cached).Length != expectedBytes)
            {
                try
                {
                    File.Delete(cached);
                }
                catch (Exception e)
                {
                    error = $"Stale cached model '{cached}' cannot be deleted: {e.Message}";
                    return false;
                }
                return false;
            }
            try
            {
                onReady(Model.Open(cached, prefault));
                return true;
            }
            catch (Exception e)
            {
                try
                {
                    File.Delete(cached);
                }
                catch (Exception deleteError)
                {
                    error = $"Cached model '{cached}' is corrupt ({e.Message}) and cannot be deleted: {deleteError.Message}";
                    return false;
                }
                return false;
            }
        }

        static string WriteCacheAtomically(string cached, byte[] bytes)
        {
            var tmp = cached + ".tmp";
            try
            {
                var directory = Path.GetDirectoryName(cached);
                if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
                File.WriteAllBytes(tmp, bytes);
                if (File.Exists(cached)) File.Delete(cached);
                File.Move(tmp, cached);
                return null;
            }
            catch (Exception e)
            {
                try
                {
                    if (File.Exists(tmp)) File.Delete(tmp);
                }
                catch (Exception cleanupError)
                {
                    return $"Could not cache model to '{cached}': {e.Message} (cleanup also failed: {cleanupError.Message})";
                }
                return $"Could not cache model to '{cached}': {e.Message}";
            }
        }

        static void FromBytes(byte[] bytes, string source, Action<Model> onReady, Action<string> onError)
        {
            Model model = null;
            string error = null;
            try
            {
                model = Model.FromBytes(bytes);
            }
            catch (Exception e)
            {
                error = $"Downloaded model '{source}' is not a usable archive: {e.Message}";
            }
            if (error != null) Fail(error, onError);
            else onReady(model);
        }

        static void Fail(string message, Action<string> onError)
        {
            if (onError != null) onError(message);
            else Debug.LogError($"[Needle] {message}");
        }
    }
}
