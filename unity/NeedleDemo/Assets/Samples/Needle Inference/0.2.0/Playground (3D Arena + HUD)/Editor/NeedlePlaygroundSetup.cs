using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace Needle.Samples.Playground.Editor
{
    public static class NeedlePlaygroundSetup
    {
        [MenuItem("Tools/Needle/Rebuild Playground Scene")]
        public static void CreateAndSave()
        {
            string root = ResolveSampleRoot();
            string texturesDir = root + "/Textures";
            string materialsDir = root + "/Materials";
            string scenesDir = root + "/Scenes";
            string scenePath = scenesDir + "/NeedlePlayground.unity";
            string gridTexturePath = texturesDir + "/ArenaGrid.png";

            EnsureFolder(texturesDir);
            EnsureFolder(materialsDir);
            EnsureFolder(scenesDir);

            var litShader = GraphicsSettings.currentRenderPipeline != null
                ? Shader.Find("Universal Render Pipeline/Lit")
                : Shader.Find("Standard");
            if (litShader == null)
                litShader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");

            Texture2D gridTex = CreateOrLoadArenaTexture(gridTexturePath);
            var mFloor = CreateOrUpdateMaterial(materialsDir + "/M_Floor.mat", litShader, new Color(0.95f, 0.98f, 1f), 0.25f, 0.1f, false, Color.black, gridTex, new Vector2(6f, 6f));
            var mTrim = CreateOrUpdateMaterial(materialsDir + "/M_FloorTrim.mat", litShader, new Color(0.08f, 0.11f, 0.17f), 0.35f, 0.3f, false, Color.black, null, Vector2.one);
            var mAccentRing = CreateOrUpdateMaterial(materialsDir + "/M_AccentRing.mat", litShader, new Color(0.14f, 0.45f, 0.72f), 0.4f, 0.2f, true, new Color(0.06f, 0.24f, 0.42f), null, Vector2.one);
            var mCommander = CreateOrUpdateMaterial(materialsDir + "/M_Commander.mat", litShader, new Color(0.18f, 0.74f, 0.98f), 0.5f, 0.25f, true, new Color(0.05f, 0.26f, 0.42f), null, Vector2.one);
            var mLyra = CreateOrUpdateMaterial(materialsDir + "/M_Lyra.mat", litShader, new Color(0.22f, 0.86f, 0.62f), 0.45f, 0.2f, true, new Color(0.04f, 0.26f, 0.18f), null, Vector2.one);
            var mKael = CreateOrUpdateMaterial(materialsDir + "/M_Kael.mat", litShader, new Color(0.36f, 0.56f, 0.96f), 0.45f, 0.25f, true, new Color(0.08f, 0.16f, 0.36f), null, Vector2.one);
            var mVex = CreateOrUpdateMaterial(materialsDir + "/M_Vex.mat", litShader, new Color(0.64f, 0.44f, 0.96f), 0.45f, 0.2f, true, new Color(0.18f, 0.10f, 0.34f), null, Vector2.one);
            var mMira = CreateOrUpdateMaterial(materialsDir + "/M_Mira.mat", litShader, new Color(0.95f, 0.72f, 0.24f), 0.5f, 0.2f, true, new Color(0.28f, 0.18f, 0.04f), null, Vector2.one);
            var mOrc = CreateOrUpdateMaterial(materialsDir + "/M_Orc.mat", litShader, new Color(0.82f, 0.22f, 0.24f), 0.4f, 0.15f, false, Color.black, null, Vector2.one);
            var mStone = CreateOrUpdateMaterial(materialsDir + "/M_Stone.mat", litShader, new Color(0.22f, 0.26f, 0.34f), 0.25f, 0.15f, false, Color.black, null, Vector2.one);
            var mTimber = CreateOrUpdateMaterial(materialsDir + "/M_Timber.mat", litShader, new Color(0.34f, 0.24f, 0.17f), 0.2f, 0.05f, false, Color.black, null, Vector2.one);
            var mSteel = CreateOrUpdateMaterial(materialsDir + "/M_Steel.mat", litShader, new Color(0.62f, 0.68f, 0.78f), 0.7f, 0.8f, false, Color.black, null, Vector2.one);
            var mLamp = CreateOrUpdateMaterial(materialsDir + "/M_Lamp.mat", litShader, new Color(1.0f, 0.8f, 0.35f), 0.6f, 0.1f, true, new Color(0.95f, 0.72f, 0.25f), null, Vector2.one);
            var mGate = CreateOrUpdateMaterial(materialsDir + "/M_Gate.mat", litShader, new Color(0.2f, 0.9f, 0.5f), 0.7f, 0.2f, true, new Color(0.12f, 0.6f, 0.32f), null, Vector2.one);
            var mOrb = CreateOrUpdateMaterial(materialsDir + "/M_Orb.mat", litShader, new Color(0.28f, 0.30f, 0.42f), 0.8f, 0.3f, true, new Color(0.16f, 0.20f, 0.38f), null, Vector2.one);
            var mWaypoint = CreateOrUpdateMaterial(materialsDir + "/M_Waypoint.mat", litShader, new Color(0.15f, 0.92f, 1.0f), 0.7f, 0.2f, true, new Color(0.12f, 0.85f, 1.0f), null, Vector2.one);
            var mBeaconRed = CreateOrUpdateMaterial(materialsDir + "/M_BeaconRed.mat", litShader, new Color(0.95f, 0.24f, 0.28f), 0.6f, 0.2f, true, new Color(0.9f, 0.18f, 0.2f), null, Vector2.one);

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.26f, 0.33f, 0.46f);
            RenderSettings.ambientEquatorColor = new Color(0.16f, 0.20f, 0.28f);
            RenderSettings.ambientGroundColor = new Color(0.08f, 0.10f, 0.14f);
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.Linear;
            RenderSettings.fogColor = new Color(0.05f, 0.07f, 0.11f);
            RenderSettings.fogStartDistance = 42f;
            RenderSettings.fogEndDistance = 105f;

            var sunGo = new GameObject("SunLight");
            sunGo.transform.rotation = Quaternion.Euler(52f, -32f, 0f);
            var sunLight = sunGo.AddComponent<Light>();
            sunLight.type = LightType.Directional;
            sunLight.color = new Color(1.0f, 0.96f, 0.90f);
            sunLight.intensity = 1.65f;
            sunLight.shadows = LightShadows.Soft;

            var rimGo = new GameObject("FillLight");
            rimGo.transform.rotation = Quaternion.Euler(28f, 148f, 0f);
            var rimLight = rimGo.AddComponent<Light>();
            rimLight.type = LightType.Directional;
            rimLight.color = new Color(0.55f, 0.72f, 0.98f);
            rimLight.intensity = 0.45f;
            rimLight.shadows = LightShadows.None;

            var camGo = new GameObject("DemoCamera");
            camGo.tag = "MainCamera";
            camGo.transform.position = new Vector3(0f, 30.37f, -30.13f);
            camGo.transform.LookAt(new Vector3(0f, 0.5f, 0.8f));
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.05f, 0.07f, 0.11f);
            cam.fieldOfView = 45f;
            cam.nearClipPlane = 0.3f;
            cam.farClipPlane = 180f;
            camGo.AddComponent<AudioListener>();

            BuildArenaEnvironment(mFloor, mTrim, mAccentRing, mStone);

            var squadGroup = new GameObject("SquadGroup");
            var squadRoot = new GameObject("SquadRoot").transform;
            squadRoot.SetParent(squadGroup.transform, false);
            squadRoot.position = new Vector3(0f, 0f, -2f);

            var commander = CreateCharacterUnit("Commander", squadGroup.transform, new Vector3(0f, 0f, -0.7f), mCommander, mSteel, 1.15f, true, false);
            var lyra = CreateCharacterUnit("Lyra", squadGroup.transform, new Vector3(-1.4f, 0f, -2.3f), mLyra, mSteel, 1.0f, false, false);
            var kael = CreateCharacterUnit("Kael", squadGroup.transform, new Vector3(1.4f, 0f, -2.3f), mKael, mSteel, 1.0f, false, false);
            var vex = CreateCharacterUnit("Vex", squadGroup.transform, new Vector3(0f, 0f, -3.6f), mVex, mSteel, 1.0f, false, false);

            var mira = CreateCharacterUnit("MiraPatrol", null, new Vector3(-5.2f, 0f, 5.5f), mMira, mSteel, 1.05f, false, false);

            var orc = CreateOrcBoss(new Vector3(10f, 0f, 6f), mOrc, mStone, mSteel, out Renderer orcBodyRenderer);
            var watchtower = CreateWatchtower(new Vector3(12f, 0f, 9f), mStone, mTimber, mBeaconRed);

            Transform campGate;
            Light[] campLights;
            Renderer[] campLampRenderers;
            BuildCampOutpost(new Vector3(-10f, 0f, -7f), mStone, mTimber, mGate, mLamp, out campGate, out campLights, out campLampRenderers);

            Transform musicOrb;
            Renderer musicOrbRenderer;
            Light musicOrbLight;
            BuildMusicPedestal(new Vector3(-9f, 0f, 5.5f), mStone, mAccentRing, mOrb, out musicOrb, out musicOrbRenderer, out musicOrbLight);

            Transform waypointMarker;
            Transform waypointRing;
            BuildWaypointMarker(mWaypoint, out waypointMarker, out waypointRing);

            var rigGo = new GameObject("NeedleDemoRig");
            var audioSource = rigGo.AddComponent<AudioSource>();
            audioSource.playOnAwake = false;
            audioSource.spatialBlend = 0f;

            var controller = rigGo.AddComponent<NeedlePlaygroundController>();
            controller.SquadRoot = squadRoot;
            controller.SquadMembers = new Transform[] { commander, lyra, kael, vex };
            controller.WaypointMarker = waypointMarker;
            controller.WaypointRing = waypointRing;
            controller.MiraPatrol = mira;
            controller.OrcChieftain = orc;
            controller.OrcBodyRenderer = orcBodyRenderer;
            controller.Watchtower = watchtower;
            controller.CampGate = campGate;
            controller.CampLights = campLights;
            controller.CampLampRenderers = campLampRenderers;
            controller.SunLight = sunLight;
            controller.MusicOrbLight = musicOrbLight;
            controller.MusicOrb = musicOrb;
            controller.MusicOrbRenderer = musicOrbRenderer;
            controller.DemoCamera = cam;
            controller.VoiceAudio = audioSource;
            controller.ModelFile = "needle3.cact";
            controller.WhistleFile = "whistle.cact";
            controller.MoveSpeed = 7.5f;
            controller.MinConfidence = 0.08f;
            controller.Threads = 4;
            controller.AutoDemo = true;

            EditorSceneManager.SaveScene(scene, scenePath);
            EditorBuildSettings.scenes = new EditorBuildSettingsScene[]
            {
                new EditorBuildSettingsScene(scenePath, true)
            };
            PlayerSettings.runInBackground = true;
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        static string ResolveSampleRoot()
        {
            string[] guids = AssetDatabase.FindAssets("t:MonoScript NeedlePlaygroundController");
            for (int i = 0; i < guids.Length; i++)
            {
                string path = AssetDatabase.GUIDToAssetPath(guids[i]).Replace('\\', '/');
                int idx = path.LastIndexOf("/Scripts/NeedlePlaygroundController.cs", System.StringComparison.OrdinalIgnoreCase);
                if (idx > 0) return path.Substring(0, idx);
            }
            return "Assets";
        }

        static void EnsureFolder(string assetPath)
        {
            if (AssetDatabase.IsValidFolder(assetPath)) return;
            string[] parts = assetPath.Split('/');
            string current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
        }

        static Texture2D CreateOrLoadArenaTexture(string gridTexturePath)
        {
            const int size = 512;
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
            var pixels = new Color32[size * size];

            var bg = new Color32(16, 21, 31, 255);
            var minor = new Color32(28, 40, 58, 255);
            var major = new Color32(46, 74, 108, 255);
            var accent = new Color32(68, 132, 184, 255);

            for (int y = 0; y < size; y++)
            {
                bool yMaj = (y % 128) < 3;
                bool yMin = (y % 32) == 0;
                bool yCenter = y == size / 2;
                for (int x = 0; x < size; x++)
                {
                    bool xMaj = (x % 128) < 3;
                    bool xMin = (x % 32) == 0;
                    bool xCenter = x == size / 2;

                    Color32 c = bg;
                    if (xMin || yMin) c = minor;
                    if (xMaj || yMaj) c = major;
                    if (xCenter || yCenter) c = accent;
                    pixels[y * size + x] = c;
                }
            }

            tex.SetPixels32(pixels);
            tex.Apply();
            File.WriteAllBytes(gridTexturePath, tex.EncodeToPNG());
            Object.DestroyImmediate(tex);

            AssetDatabase.ImportAsset(gridTexturePath, ImportAssetOptions.ForceUpdate);
            var importer = AssetImporter.GetAtPath(gridTexturePath) as TextureImporter;
            if (importer != null)
            {
                importer.sRGBTexture = true;
                importer.wrapMode = TextureWrapMode.Repeat;
                importer.filterMode = FilterMode.Bilinear;
                importer.anisoLevel = 8;
                importer.mipmapEnabled = true;
                importer.SaveAndReimport();
            }

            return AssetDatabase.LoadAssetAtPath<Texture2D>(gridTexturePath);
        }

        static Material CreateOrUpdateMaterial(
            string path,
            Shader shader,
            Color baseColor,
            float smoothness,
            float metallic,
            bool emission,
            Color emissionColor,
            Texture2D mainTex,
            Vector2 texScale)
        {
            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat == null)
            {
                mat = new Material(shader);
                AssetDatabase.CreateAsset(mat, path);
            }
            else
            {
                mat.shader = shader;
            }

            mat.color = baseColor;
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", baseColor);
            if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", smoothness);
            if (mat.HasProperty("_Metallic")) mat.SetFloat("_Metallic", metallic);
            if (mainTex != null)
            {
                mat.mainTexture = mainTex;
                mat.mainTextureScale = texScale;
                if (mat.HasProperty("_BaseMap"))
                {
                    mat.SetTexture("_BaseMap", mainTex);
                    mat.SetTextureScale("_BaseMap", texScale);
                }
            }

            if (emission)
            {
                mat.EnableKeyword("_EMISSION");
                if (mat.HasProperty("_EmissionColor")) mat.SetColor("_EmissionColor", emissionColor);
            }
            else
            {
                mat.DisableKeyword("_EMISSION");
            }

            EditorUtility.SetDirty(mat);
            return mat;
        }

        static GameObject CreateBox(string name, Transform parent, Vector3 localPos, Vector3 scale, Material mat)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = scale;
            go.GetComponent<Renderer>().sharedMaterial = mat;
            return go;
        }

        static GameObject CreateCylinder(string name, Transform parent, Vector3 localPos, Vector3 scale, Material mat)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = scale;
            go.GetComponent<Renderer>().sharedMaterial = mat;
            return go;
        }

        static GameObject CreateSphere(string name, Transform parent, Vector3 localPos, Vector3 scale, Material mat)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = scale;
            go.GetComponent<Renderer>().sharedMaterial = mat;
            return go;
        }

        static GameObject CreateCapsule(string name, Transform parent, Vector3 localPos, Vector3 scale, Material mat)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = scale;
            go.GetComponent<Renderer>().sharedMaterial = mat;
            return go;
        }

        static void BuildArenaEnvironment(Material mFloor, Material mTrim, Material mAccentRing, Material mStone)
        {
            var env = new GameObject("Environment").transform;

            CreateBox("ArenaFloor", env, new Vector3(0f, -0.1f, 0f), new Vector3(36f, 0.2f, 36f), mFloor);

            CreateBox("CurbNorth", env, new Vector3(0f, 0.15f, 18.25f), new Vector3(37f, 0.3f, 0.5f), mTrim);
            CreateBox("CurbSouth", env, new Vector3(0f, 0.15f, -18.25f), new Vector3(37f, 0.3f, 0.5f), mTrim);
            CreateBox("CurbEast", env, new Vector3(18.25f, 0.15f, 0f), new Vector3(0.5f, 0.3f, 36f), mTrim);
            CreateBox("CurbWest", env, new Vector3(-18.25f, 0.15f, 0f), new Vector3(0.5f, 0.3f, 36f), mTrim);

            CreateCylinder("CenterPad", env, new Vector3(0f, 0.015f, 0f), new Vector3(4.2f, 0.03f, 4.2f), mTrim);
            CreateCylinder("CenterRing", env, new Vector3(0f, 0.032f, 0f), new Vector3(3.2f, 0.02f, 3.2f), mAccentRing);

            Vector3[] corners = new Vector3[]
            {
                new Vector3(-16.5f, 0f, -16.5f),
                new Vector3(16.5f, 0f, -16.5f),
                new Vector3(-16.5f, 0f, 16.5f),
                new Vector3(16.5f, 0f, 16.5f),
            };
            for (int i = 0; i < corners.Length; i++)
            {
                var p = corners[i];
                CreateBox("CornerPylon_" + i, env, p + new Vector3(0f, 0.9f, 0f), new Vector3(0.9f, 1.8f, 0.9f), mStone);
                CreateBox("CornerCap_" + i, env, p + new Vector3(0f, 1.95f, 0f), new Vector3(0.6f, 0.25f, 0.6f), mAccentRing);
            }
        }

        static Transform CreateCharacterUnit(
            string name,
            Transform parent,
            Vector3 worldPos,
            Material bodyMat,
            Material trimMat,
            float scale,
            bool isCommander,
            bool isBoss)
        {
            var root = new GameObject(name).transform;
            if (parent != null) root.SetParent(parent, false);
            root.position = worldPos;

            CreateCylinder("BaseDisc", root, new Vector3(0f, 0.04f, 0f), new Vector3(1.1f * scale, 0.04f, 1.1f * scale), bodyMat);
            CreateCapsule("Torso", root, new Vector3(0f, 0.72f * scale, 0f), new Vector3(0.56f * scale, 0.56f * scale, 0.56f * scale), bodyMat);
            CreateSphere("Head", root, new Vector3(0f, 1.42f * scale, 0f), new Vector3(0.42f * scale, 0.42f * scale, 0.42f * scale), bodyMat);
            CreateBox("Visor", root, new Vector3(0f, 1.44f * scale, 0.16f * scale), new Vector3(0.32f * scale, 0.12f * scale, 0.16f * scale), trimMat);

            if (isCommander)
            {
                CreateBox("PauldronL", root, new Vector3(-0.36f * scale, 1.05f * scale, 0f), new Vector3(0.24f * scale, 0.22f * scale, 0.34f * scale), trimMat);
                CreateBox("PauldronR", root, new Vector3(0.36f * scale, 1.05f * scale, 0f), new Vector3(0.24f * scale, 0.22f * scale, 0.34f * scale), trimMat);
                CreateBox("Crest", root, new Vector3(0f, 1.70f * scale, -0.02f), new Vector3(0.10f * scale, 0.22f * scale, 0.34f * scale), trimMat);
            }
            else if (isBoss)
            {
                CreateBox("ArmorL", root, new Vector3(-0.46f * scale, 1.05f * scale, 0f), new Vector3(0.32f * scale, 0.32f * scale, 0.44f * scale), trimMat);
                CreateBox("ArmorR", root, new Vector3(0.46f * scale, 1.05f * scale, 0f), new Vector3(0.32f * scale, 0.32f * scale, 0.44f * scale), trimMat);
            }
            else
            {
                CreateBox("Backpack", root, new Vector3(0f, 0.85f * scale, -0.26f * scale), new Vector3(0.34f * scale, 0.42f * scale, 0.18f * scale), trimMat);
            }

            return root;
        }

        static Transform CreateOrcBoss(Vector3 worldPos, Material mOrc, Material mStone, Material mSteel, out Renderer torsoRenderer)
        {
            var root = new GameObject("OrcChieftain").transform;
            root.position = worldPos;
            root.rotation = Quaternion.LookRotation(new Vector3(-1f, 0f, -0.6f).normalized, Vector3.up);

            CreateCylinder("OrcRing", root, new Vector3(0f, 0.04f, 0f), new Vector3(2.2f, 0.04f, 2.2f), mStone);
            var torso = CreateCapsule("OrcTorso", root, new Vector3(0f, 1.05f, 0f), new Vector3(0.95f, 0.82f, 0.85f), mOrc);
            torsoRenderer = torso.GetComponent<Renderer>();
            CreateSphere("OrcHead", root, new Vector3(0f, 1.98f, 0.05f), new Vector3(0.58f, 0.56f, 0.58f), mOrc);
            CreateBox("OrcPauldronL", root, new Vector3(-0.62f, 1.48f, 0f), new Vector3(0.42f, 0.38f, 0.52f), mStone);
            CreateBox("OrcPauldronR", root, new Vector3(0.62f, 1.48f, 0f), new Vector3(0.42f, 0.38f, 0.52f), mStone);
            CreateBox("OrcCleaver", root, new Vector3(0.68f, 1.15f, 0.52f), new Vector3(0.14f, 1.45f, 0.36f), mSteel);
            CreateBox("OrcHornL", root, new Vector3(-0.22f, 2.24f, 0.14f), new Vector3(0.10f, 0.28f, 0.10f), mSteel);
            CreateBox("OrcHornR", root, new Vector3(0.22f, 2.24f, 0.14f), new Vector3(0.10f, 0.28f, 0.10f), mSteel);

            return root;
        }

        static Transform CreateWatchtower(Vector3 worldPos, Material mStone, Material mTimber, Material mBeaconRed)
        {
            var root = new GameObject("Watchtower").transform;
            root.position = worldPos;

            CreateBox("BasePlinth", root, new Vector3(0f, 0.35f, 0f), new Vector3(3.2f, 0.7f, 3.2f), mStone);
            CreateBox("TowerShaft", root, new Vector3(0f, 1.85f, 0f), new Vector3(2.1f, 2.5f, 2.1f), mStone);
            CreateBox("UpperDeck", root, new Vector3(0f, 3.25f, 0f), new Vector3(2.9f, 0.32f, 2.9f), mTimber);
            CreateBox("BattlementN", root, new Vector3(0f, 3.62f, 1.3f), new Vector3(2.9f, 0.42f, 0.25f), mStone);
            CreateBox("BattlementS", root, new Vector3(0f, 3.62f, -1.3f), new Vector3(2.9f, 0.42f, 0.25f), mStone);
            CreateBox("BattlementE", root, new Vector3(1.3f, 3.62f, 0f), new Vector3(0.25f, 0.42f, 2.4f), mStone);
            CreateBox("BattlementW", root, new Vector3(-1.3f, 3.62f, 0f), new Vector3(0.25f, 0.42f, 2.4f), mStone);
            CreateSphere("SignalBeacon", root, new Vector3(0f, 4.0f, 0f), new Vector3(0.55f, 0.55f, 0.55f), mBeaconRed);

            return root;
        }

        static void BuildCampOutpost(
            Vector3 worldPos,
            Material mStone,
            Material mTimber,
            Material mGate,
            Material mLamp,
            out Transform campGate,
            out Light[] campLights,
            out Renderer[] campLampRenderers)
        {
            var root = new GameObject("CampOutpost").transform;
            root.position = worldPos;

            CreateBox("OutpostDeck", root, new Vector3(0f, 0.1f, 0f), new Vector3(6.2f, 0.2f, 4.4f), mStone);
            CreateBox("ShelterBody", root, new Vector3(0f, 1.15f, -0.95f), new Vector3(4.6f, 2.0f, 2.0f), mTimber);
            CreateBox("ShelterRoof", root, new Vector3(0f, 2.3f, -0.95f), new Vector3(5.1f, 0.35f, 2.5f), mStone);

            CreateBox("GatePillarL", root, new Vector3(-1.6f, 1.25f, 1.4f), new Vector3(0.55f, 2.3f, 0.55f), mStone);
            CreateBox("GatePillarR", root, new Vector3(1.6f, 1.25f, 1.4f), new Vector3(0.55f, 2.3f, 0.55f), mStone);
            CreateBox("GateLintel", root, new Vector3(0f, 2.42f, 1.4f), new Vector3(3.8f, 0.35f, 0.65f), mStone);

            var gateGo = CreateBox("CampGate", root, new Vector3(0f, 0.12f, 1.4f), new Vector3(2.65f, 0.28f, 0.22f), mGate);
            campGate = gateGo.transform;

            campLights = new Light[2];
            campLampRenderers = new Renderer[2];
            float[] lampXs = new float[] { -2.45f, 2.45f };

            for (int i = 0; i < 2; i++)
            {
                var post = new GameObject("CampLampPost_" + i).transform;
                post.SetParent(root, false);
                post.localPosition = new Vector3(lampXs[i], 0f, 1.95f);

                CreateCylinder("Pole", post, new Vector3(0f, 1.1f, 0f), new Vector3(0.16f, 1.0f, 0.16f), mStone);
                var bulb = CreateSphere("LampBulb_" + i, post, new Vector3(0f, 2.25f, 0f), new Vector3(0.45f, 0.45f, 0.45f), mLamp);
                campLampRenderers[i] = bulb.GetComponent<Renderer>();

                var lightGo = new GameObject("CampLight_" + i);
                lightGo.transform.SetParent(post, false);
                lightGo.transform.localPosition = new Vector3(0f, 2.55f, 0f);
                var pointLight = lightGo.AddComponent<Light>();
                pointLight.type = LightType.Point;
                pointLight.range = 14f;
                pointLight.color = new Color(1.0f, 0.84f, 0.46f);
                pointLight.intensity = 24f;
                campLights[i] = pointLight;
            }
        }

        static void BuildMusicPedestal(
            Vector3 worldPos,
            Material mStone,
            Material mAccentRing,
            Material mOrb,
            out Transform musicOrb,
            out Renderer musicOrbRenderer,
            out Light musicOrbLight)
        {
            var root = new GameObject("MusicPedestal").transform;
            root.position = worldPos;

            CreateCylinder("PatrolRingPad", root, new Vector3(0f, 0.02f, 0f), new Vector3(7.9f, 0.02f, 7.9f), mAccentRing);
            CreateCylinder("PatrolInnerFloor", root, new Vector3(0f, 0.026f, 0f), new Vector3(7.45f, 0.02f, 7.45f), mStone);
            CreateCylinder("PedestalBase", root, new Vector3(0f, 0.3f, 0f), new Vector3(1.5f, 0.3f, 1.5f), mStone);
            CreateCylinder("PedestalColumn", root, new Vector3(0f, 0.72f, 0f), new Vector3(0.85f, 0.4f, 0.85f), mStone);

            var orbGo = CreateSphere("MusicOrb", root, new Vector3(0f, 1.6f, 0f), new Vector3(0.74f, 0.74f, 0.74f), mOrb);
            musicOrb = orbGo.transform;
            musicOrbRenderer = orbGo.GetComponent<Renderer>();

            var orbLightGo = new GameObject("MusicOrbLight");
            orbLightGo.transform.SetParent(musicOrb, false);
            orbLightGo.transform.localPosition = new Vector3(0f, 0.25f, 0f);
            musicOrbLight = orbLightGo.AddComponent<Light>();
            musicOrbLight.type = LightType.Point;
            musicOrbLight.range = 14f;
            musicOrbLight.color = new Color(0.5f, 0.8f, 1.0f);
            musicOrbLight.intensity = 18f;
            musicOrbLight.enabled = false;
        }

        static void BuildWaypointMarker(Material mWaypoint, out Transform waypointMarker, out Transform waypointRing)
        {
            var root = new GameObject("WaypointMarker").transform;
            root.position = Vector3.zero;

            var ring = CreateCylinder("WaypointRing", root, new Vector3(0f, 0.04f, 0f), new Vector3(1.8f, 0.05f, 1.8f), mWaypoint);
            waypointRing = ring.transform;

            var spire = CreateBox("WaypointCrystal", root, new Vector3(0f, 0.75f, 0f), new Vector3(0.24f, 0.9f, 0.24f), mWaypoint);
            spire.transform.localRotation = Quaternion.Euler(0f, 45f, 0f);

            root.gameObject.SetActive(false);
            waypointMarker = root;
        }
    }
}
