using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace Needle
{
    public static class ModelLoader
    {
        public static string StreamingPath(string fileName) =>
            Path.Combine(Application.streamingAssetsPath, fileName);

        public static bool IsApkUrl(string path) =>
            path.StartsWith("jar:file://", StringComparison.Ordinal) ||
            path.IndexOf("!/assets", StringComparison.Ordinal) >= 0;

        public static bool CanOpenDirect(string fileName)
        {
            var path = StreamingPath(fileName);
            return !IsApkUrl(path) && File.Exists(path);
        }

        public static string CachedPath(string fileName) =>
            Path.Combine(Application.persistentDataPath, fileName);

        public static Model OpenDirect(string absolutePath, bool prefault = true)
        {
            if (IsApkUrl(absolutePath))
                throw new IOException(
                    $"Model '{absolutePath}' is inside the APK and is not a file. " +
                    $"Load it with ModelLoader.OpenAsync instead.");
            return Model.Open(absolutePath, prefault);
        }

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
