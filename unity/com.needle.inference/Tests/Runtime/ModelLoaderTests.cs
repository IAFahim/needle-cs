using System.Collections;
using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace Needle.Unity.Tests
{
    public sealed class ModelLoaderTests
    {
        // ModelLoader.OpenAsync only yields while it downloads from inside an APK. Everywhere
        // else it finishes in its first step, so EditMode tests can run it to the end directly.
        static void RunToEnd(IEnumerator routine)
        {
            int steps = 0;
            while (routine.MoveNext())
            {
                Assert.IsNull(routine.Current, "OpenAsync waited on a web request outside an APK.");
                Assert.Less(++steps, 1000, "OpenAsync did not finish.");
            }
        }

        [Test]
        public void RecognisesApkPaths()
        {
            Assert.IsTrue(ModelLoader.IsApkUrl("jar:file:///data/app/com.game/base.apk!/assets/needle3.cact"));
            Assert.IsTrue(ModelLoader.IsApkUrl("/data/app/com.game/base.apk!/assets/needle3.cact"));
            Assert.IsFalse(ModelLoader.IsApkUrl("/home/player/Game_Data/StreamingAssets/needle3.cact"));
            Assert.IsFalse(ModelLoader.IsApkUrl(@"C:\Games\Game_Data\StreamingAssets\needle3.cact"));
        }

        [Test]
        public void PathsLiveUnderUnityFolders()
        {
            Assert.AreEqual(Path.Combine(Application.streamingAssetsPath, "needle3.cact"), ModelLoader.StreamingPath("needle3.cact"));
            Assert.AreEqual(Path.Combine(Application.persistentDataPath, "needle3.cact"), ModelLoader.CachedPath("needle3.cact"));
        }

        [Test]
        public void OpenDirectRefusesApkPaths()
        {
            Assert.Throws<IOException>(() => ModelLoader.OpenDirect("jar:file:///data/app/base.apk!/assets/needle3.cact"));
        }

        [Test]
        public void OpenAsyncReportsAMissingFileName()
        {
            string error = null;
            bool ready = false;
            RunToEnd(ModelLoader.OpenAsync("", m => ready = true, e => error = e));
            Assert.IsFalse(ready);
            StringAssert.Contains("StreamingAssets file name", error);
        }

        [Test]
        public void OpenAsyncReportsAMissingCallback()
        {
            string error = null;
            RunToEnd(ModelLoader.OpenAsync("needle3.cact", null, e => error = e));
            StringAssert.Contains("onReady", error);
        }

        [Test]
        public void OpenAsyncReportsAMissingFile()
        {
            string error = null;
            bool ready = false;
            RunToEnd(ModelLoader.OpenAsync("needle-test-does-not-exist.cact", m => ready = true, e => error = e));
            Assert.IsFalse(ready);
            StringAssert.Contains("Could not open model", error);
        }

        [Test]
        public void OpenAsyncReportsAFileThatIsNotAModel()
        {
            string path = Path.Combine(Path.GetTempPath(), "needle-test-not-a-model.cact");
            File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            try
            {
                string error = null;
                bool ready = false;
                RunToEnd(ModelLoader.OpenAsync(path, m => { ready = true; m.Dispose(); }, e => error = e));
                Assert.IsFalse(ready, "A eight-byte file opened as a model.");
                StringAssert.Contains("Could not open model", error);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Test]
        public void OpenAsyncOpensARealModel()
        {
            // Path.Combine keeps an absolute second argument, so an absolute path stands in for a
            // StreamingAssets file name here.
            string path = NeedleTestAssets.RequireModel();
            Model model = null;
            string error = null;
            RunToEnd(ModelLoader.OpenAsync(path, m => model = m, e => error = e, cache: false, prefault: false));
            Assert.IsNull(error, error);
            Assert.IsNotNull(model);
            model.Dispose();
        }
    }
}
