using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Needle;
using Needle.Tools;
using UnityEngine;
using UnityEngine.Rendering;

namespace Needle.Samples.Playground
{
    public sealed unsafe class NeedlePlaygroundController : MonoBehaviour
    {
        public Transform SquadRoot;
        public Transform[] SquadMembers;
        public Transform WaypointMarker;
        public Transform WaypointRing;
        public Transform MiraPatrol;
        public Transform OrcChieftain;
        public Renderer OrcBodyRenderer;
        public Transform Watchtower;
        public Transform CampGate;
        public Light[] CampLights;
        public Renderer[] CampLampRenderers;
        public Light SunLight;
        public Light MusicOrbLight;
        public Transform MusicOrb;
        public Renderer MusicOrbRenderer;
        public Camera DemoCamera;
        public AudioSource VoiceAudio;

        public string ModelFile = "needle3.cact";
        public string WhistleFile = "whistle.cact";
        public float MoveSpeed = 7.5f;
        public float MinConfidence = 0.3f;
        public int Threads = 4;
        public bool AutoDemo = true;

        const float ArenaHalfExtent = 16f;
        const string InputControlName = "NeedleOrderInput";

        const string AssistantToolsJson =
            "[{\"name\":\"get_weather\",\"description\":\"Get the current weather for a city.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\",\"description\":\"City name\"}},\"required\":[\"city\"]}}," +
            "{\"name\":\"set_timer\",\"description\":\"Set a countdown timer.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"minutes\":{\"type\":\"integer\",\"description\":\"Length in minutes\"}},\"required\":[\"minutes\"]}}," +
            "{\"name\":\"play_music\",\"description\":\"Play a song or artist.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\",\"description\":\"Song or artist\"}},\"required\":[\"query\"]}}]";

        const string SmartCampToolsJson =
            "[{\"name\":\"set_lights\",\"description\":\"Turn lights on or off or set brightness for a room or camp area.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"room\":{\"type\":\"string\",\"description\":\"Room or area name\"},\"on\":{\"type\":\"boolean\",\"description\":\"Whether light is on\"},\"brightness\":{\"type\":\"integer\",\"description\":\"Brightness percentage 0 to 100\"}},\"required\":[\"room\",\"on\"]}}," +
            "{\"name\":\"lock_doors\",\"description\":\"Lock or unlock the camp gate and doors.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"locked\":{\"type\":\"boolean\",\"description\":\"True to lock, false to unlock\"}},\"required\":[\"locked\"]}}]";

        readonly object stateGate = new object();
        readonly byte[] submitBuffer = new byte[ChannelLimits.QueryCapacity];
        readonly byte[] decodeBuffer = new byte[4096];
        readonly List<string> history = new List<string>(24);
        readonly Dictionary<string, float[]> cachedWav = new Dictionary<string, float[]>(4);
        readonly Dictionary<string, AudioClip> cachedClips = new Dictionary<string, AudioClip>(4);

        int activeMode;
        int selectedThreads = 4;
        int initGeneration;
        bool isDisposed;

        NeedleHost squadHost;
        NeedleChannel squadChannel;
        volatile bool squadReady;
        volatile bool squadLoading;

        NeedleHost campHost;
        NeedleChannel campChannel;
        volatile bool campReady;
        volatile bool campLoading;

        VoiceHost voiceHost;
        VoiceChannel voiceChannel;
        volatile bool voiceReady;
        volatile bool voiceLoading;

        volatile string initError = "";

        ulong pendingTicket;
        int pendingMode = -1;
        bool hasPending;
        double queryStartTime;
        bool autoDemoSent;
        double enableTime;

        Vector3 squadPos = new Vector3(0f, 0f, -2f);
        Vector3 squadDest = new Vector3(0f, 0f, -2f);
        Vector3 squadFacing = Vector3.forward;
        bool squadMoving;
        Formation currentFormation = Formation.Wedge;
        string followingTarget = "";
        string engagingTarget = "";
        float attackPulse;
        float givePulse;
        float waypointTimer;
        int[] inventory = new int[] { 5, 2, 2, 40, 100 };

        bool campLightsOn = true;
        int campBrightness = 85;
        string lastRoom = "camp";
        bool campLocked;
        float gateClosedAmount;

        string weatherText = "No weather query yet";
        float timerRemainingSeconds;
        string timerLabel = "None";
        string musicNowPlaying = "Idle";
        bool musicPlaying;
        float musicPulsePhase;
        bool audioEnabled = true;

        float miraAngle;
        Vector3 miraPos = new Vector3(-9f, 0f, 5f);

        bool followCamera;
        float camYaw = 0f;
        float camPitch = 44f;
        float camDistance = 43f;
        Vector3 camFocus = new Vector3(0f, 0.5f, 0.8f);
        bool draggingCamera;
        Vector2 lastMousePos;

        string inputOrder = "attack the orc chieftain";
        string lastStatus = "Starting engine...";
        string lastCalls = "—";
        string lastReasoning = "—";
        string lastTranscript = "—";
        string lastTiming = "Waiting for first call";
        long lastPollAllocBytes;
        int activeThreadId;

        Vector2 historyScroll;
        Vector2 presetScroll;

        bool stylesReady;
        Texture2D darkBgTex;
        Texture2D panelTex;
        Texture2D cardTex;
        Texture2D inputTex;
        Texture2D accentTex;
        Texture2D accentHoverTex;
        Texture2D buttonTex;
        Texture2D buttonHoverTex;
        Texture2D greenPillTex;
        Texture2D amberPillTex;
        GUIStyle panelStyle;
        GUIStyle cardStyle;
        GUIStyle titleStyle;
        GUIStyle labelStyle;
        GUIStyle mutedStyle;
        GUIStyle monoStyle;
        GUIStyle inputStyle;
        GUIStyle buttonStyle;
        GUIStyle activeButtonStyle;
        GUIStyle presetButtonStyle;
        GUIStyle nameplateStyle;

        void OnEnable()
        {
            isDisposed = false;
            selectedThreads = ClampThreads(Threads);
            activeThreadId = selectedThreads;
            enableTime = Time.realtimeSinceStartupAsDouble;
            autoDemoSent = false;

            EnsureRenderPipelineMaterials();

            try
            {
                Kernels.TryInstallSidecar();
            }
            catch (Exception e)
            {
                initError = e.Message;
            }

            StartHostForModeAsync(activeMode, selectedThreads);
            ResetArenaState();
        }

        static void EnsureRenderPipelineMaterials()
        {
            if (GraphicsSettings.currentRenderPipeline != null) return;
            Shader fallback = Shader.Find("Standard");
            if (fallback == null) return;
            var renderers = FindObjectsByType<Renderer>();
            for (int i = 0; i < renderers.Length; i++)
            {
                var mat = renderers[i].sharedMaterial;
                if (mat == null || mat.shader == null) continue;
                if (mat.shader.name != "Universal Render Pipeline/Lit") continue;
                Color baseCol = mat.HasProperty("_BaseColor") ? mat.GetColor("_BaseColor") : Color.white;
                Texture baseTex = mat.HasProperty("_BaseMap") ? mat.GetTexture("_BaseMap") : null;
                Vector2 texScale = mat.HasProperty("_BaseMap") ? mat.GetTextureScale("_BaseMap") : Vector2.one;
                mat.shader = fallback;
                mat.color = baseCol;
                if (baseTex != null)
                {
                    mat.mainTexture = baseTex;
                    mat.mainTextureScale = texScale;
                }
            }
        }

        void OnDisable()
        {
            isDisposed = true;
            lock (stateGate)
            {
                initGeneration++;
                DisposeAllHostsLocked();
            }
        }

        void OnDestroy()
        {
            foreach (var kv in cachedClips)
                if (kv.Value != null) Destroy(kv.Value);
            cachedClips.Clear();

            DestroyTex(ref darkBgTex);
            DestroyTex(ref panelTex);
            DestroyTex(ref cardTex);
            DestroyTex(ref inputTex);
            DestroyTex(ref accentTex);
            DestroyTex(ref accentHoverTex);
            DestroyTex(ref buttonTex);
            DestroyTex(ref buttonHoverTex);
            DestroyTex(ref greenPillTex);
            DestroyTex(ref amberPillTex);
        }

        static void DestroyTex(ref Texture2D tex)
        {
            if (tex != null) Destroy(tex);
            tex = null;
        }

        static int ClampThreads(int count) => Math.Max(1, Math.Min(8, count <= 0 ? 4 : count));

        void DisposeAllHostsLocked()
        {
            squadReady = false;
            squadLoading = false;
            campReady = false;
            campLoading = false;
            voiceReady = false;
            voiceLoading = false;
            hasPending = false;

            try { squadHost?.Dispose(); } catch { }
            squadHost = null;
            squadChannel = default;

            try { campHost?.Dispose(); } catch { }
            campHost = null;
            campChannel = default;

            try { voiceHost?.Dispose(); } catch { }
            voiceHost = null;
            voiceChannel = default;
        }

        void StartHostForModeAsync(int mode, int threads)
        {
            string streamingDir = Application.streamingAssetsPath;
            string needlePath = Path.Combine(streamingDir, ModelFile);
            string whistlePath = Path.Combine(streamingDir, WhistleFile);

            if (!File.Exists(needlePath))
            {
                initError = "Missing " + ModelFile + " in StreamingAssets. Copy from ref/hf/ or run tools/fetch.sh.";
                lastStatus = initError;
                return;
            }

            int gen;
            lock (stateGate)
            {
                if (mode == 0 && (squadReady || squadLoading)) return;
                if (mode == 1 && (campReady || campLoading)) return;
                if (mode == 2 && (voiceReady || voiceLoading)) return;
                gen = initGeneration;
                if (mode == 0) squadLoading = true;
                else if (mode == 1) campLoading = true;
                else voiceLoading = true;
            }

            var worker = new Thread(() =>
            {
                try
                {
                    if (mode == 0)
                    {
                        var h = new NeedleHost();
                        var ch = h.StartTyped<SquadSink, SquadCommand>(new HostOptions
                        {
                            ModelPath = needlePath,
                            ToolsJson = Encoding.UTF8.GetString(SquadCommands.Schema.ToArray()),
                            StatelessQueries = true,
                            Session = new NeedleOptions
                            {
                                Threading = ThreadPolicy.Game.WithThreads(threads),
                                RetrievalTopK = SquadCommands.GrammarToolCount,
                                RefuseBelow = MinConfidence,
                            },
                        });
                        lock (stateGate)
                        {
                            if (isDisposed || gen != initGeneration) { h.Dispose(); return; }
                            squadHost = h;
                            squadChannel = ch;
                            squadReady = true;
                            squadLoading = false;
                        }
                    }
                    else if (mode == 1)
                    {
                        var h = new NeedleHost();
                        var ch = h.Start(new HostOptions
                        {
                            ModelPath = needlePath,
                            ToolsJson = SmartCampToolsJson,
                            StatelessQueries = true,
                            Session = new NeedleOptions
                            {
                                Threading = ThreadPolicy.Game.WithThreads(threads),
                                RefuseBelow = MinConfidence,
                            },
                        });
                        lock (stateGate)
                        {
                            if (isDisposed || gen != initGeneration) { h.Dispose(); return; }
                            campHost = h;
                            campChannel = ch;
                            campReady = true;
                            campLoading = false;
                        }
                    }
                    else
                    {
                        if (!File.Exists(whistlePath)) throw new FileNotFoundException("Missing whistle model: " + whistlePath);
                        var vh = new VoiceHost();
                        var vc = vh.Start(new VoiceHostOptions
                        {
                            WhistleModelPath = whistlePath,
                            NeedleModelPath = needlePath,
                            ToolsJson = AssistantToolsJson,
                            WhistleThreads = threads,
                            Session = new NeedleOptions
                            {
                                Threading = ThreadPolicy.Game.WithThreads(threads),
                                Stateless = true,
                                RefuseBelow = MinConfidence,
                            },
                        });
                        lock (stateGate)
                        {
                            if (isDisposed || gen != initGeneration) { vh.Dispose(); return; }
                            voiceHost = vh;
                            voiceChannel = vc;
                            voiceReady = true;
                            voiceLoading = false;
                        }
                    }
                }
                catch (Exception e)
                {
                    lock (stateGate)
                    {
                        if (mode == 0) squadLoading = false;
                        else if (mode == 1) campLoading = false;
                        else voiceLoading = false;
                        initError = e.GetType().Name + ": " + e.Message;
                    }
                }
            })
            {
                IsBackground = true,
                Name = "needle-demo-init-" + mode,
            };
            worker.Start();
        }

        public void SelectMode(int mode)
        {
            if (mode < 0 || mode > 2) return;
            activeMode = mode;
            if (mode == 0) inputOrder = "switch to wedge formation and move to 8, 6";
            else if (mode == 1) inputOrder = "dim the camp lights to 25 and unlock the gate";
            else inputOrder = "what's it like in Lagos right now?";
            StartHostForModeAsync(mode, selectedThreads);
        }

        public void SetThreads(int threads)
        {
            int next = ClampThreads(threads);
            if (next == selectedThreads && activeThreadId == next) return;
            selectedThreads = next;
            activeThreadId = next;
            lock (stateGate)
            {
                initGeneration++;
                DisposeAllHostsLocked();
            }
            lastStatus = "Switching workers to " + next + " threads...";
            StartHostForModeAsync(activeMode, next);
        }

        public bool RunOrder(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return false;
            string trimmed = text.Trim();
            inputOrder = trimmed;

            int byteCount = Encoding.UTF8.GetByteCount(trimmed);
            if (byteCount > submitBuffer.Length)
            {
                lastStatus = "Prompt too long";
                return false;
            }
            int len = Encoding.UTF8.GetBytes(trimmed, 0, trimmed.Length, submitBuffer, 0);
            var utf8 = new ReadOnlySpan<byte>(submitBuffer, 0, len);

            if (activeMode == 0)
            {
                if (!squadReady || squadHost == null)
                {
                    StartHostForModeAsync(0, selectedThreads);
                    lastStatus = "Loading Squad engine...";
                    return false;
                }
                if (hasPending && pendingMode == 0) squadHost.Cancel(pendingTicket);
                if (!squadChannel.TrySubmitQuery(utf8, out ulong ticket)) return false;
                pendingTicket = ticket;
                pendingMode = 0;
                hasPending = true;
                queryStartTime = Time.realtimeSinceStartupAsDouble;
                lastStatus = "Running Squad inference...";
                return true;
            }

            if (activeMode == 1)
            {
                if (!campReady || campHost == null)
                {
                    StartHostForModeAsync(1, selectedThreads);
                    lastStatus = "Loading Smart Camp engine...";
                    return false;
                }
                if (hasPending && pendingMode == 1) campHost.Cancel(pendingTicket);
                if (!campChannel.TrySubmitQuery(utf8, out ulong ticket)) return false;
                pendingTicket = ticket;
                pendingMode = 1;
                hasPending = true;
                queryStartTime = Time.realtimeSinceStartupAsDouble;
                lastStatus = "Running Smart Camp inference...";
                return true;
            }

            if (!voiceReady || voiceHost == null)
            {
                StartHostForModeAsync(2, selectedThreads);
                lastStatus = "Loading Assistant + Whistle...";
                return false;
            }
            if (!voiceHost.Channel.TrySubmitQuery(utf8, out ulong vTicket)) return false;
            pendingTicket = vTicket;
            pendingMode = 2;
            hasPending = true;
            queryStartTime = Time.realtimeSinceStartupAsDouble;
            lastStatus = "Running Assistant inference...";
            return true;
        }

        public bool RunVoiceClip(string clipName)
        {
            if (!voiceReady || voiceHost == null)
            {
                SelectMode(2);
                lastStatus = "Loading Whistle + Assistant...";
                return false;
            }

            string path = Path.Combine(Application.streamingAssetsPath, clipName);
            if (!File.Exists(path))
            {
                lastStatus = "Clip missing: " + clipName;
                return false;
            }

            if (!cachedWav.TryGetValue(clipName, out float[] pcm))
            {
                pcm = Wav.Read(path);
                cachedWav[clipName] = pcm;
            }

            if (audioEnabled && VoiceAudio != null)
            {
                if (!cachedClips.TryGetValue(clipName, out AudioClip clip))
                {
                    clip = AudioClip.Create(clipName, pcm.Length, 1, Wav.SampleRate, false);
                    clip.SetData(pcm, 0);
                    cachedClips[clipName] = clip;
                }
                VoiceAudio.Stop();
                VoiceAudio.PlayOneShot(clip, 0.7f);
            }

            voiceChannel.WriteAudio(pcm);
            if (!voiceChannel.TryEndUtterance(out ulong utterance))
            {
                lastStatus = "Voice channel busy";
                return false;
            }
            pendingTicket = utterance;
            pendingMode = 3;
            hasPending = true;
            queryStartTime = Time.realtimeSinceStartupAsDouble;
            lastStatus = "Whistle transcribing " + clipName + "...";
            return true;
        }

        public void ResetArenaState()
        {
            squadPos = new Vector3(0f, 0f, -2f);
            squadDest = squadPos;
            squadFacing = Vector3.forward;
            squadMoving = false;
            currentFormation = Formation.Wedge;
            followingTarget = "";
            engagingTarget = "";
            attackPulse = 0f;
            givePulse = 0f;
            waypointTimer = 0f;
            inventory[0] = 5;
            inventory[1] = 2;
            inventory[2] = 2;
            inventory[3] = 40;
            inventory[4] = 100;

            campLightsOn = true;
            campBrightness = 85;
            campLocked = false;
            gateClosedAmount = 0f;
            musicPlaying = false;
            musicNowPlaying = "Idle";
            timerRemainingSeconds = 0f;
            timerLabel = "None";
            if (SunLight != null) SunLight.color = new Color(1f, 0.96f, 0.88f);
            ApplyCampVisuals();
            UpdateSquadTransforms(1f);
        }

        void Update()
        {
            if (AutoDemo && !autoDemoSent && squadReady && (Time.realtimeSinceStartupAsDouble - enableTime) > 0.45)
            {
                autoDemoSent = true;
                RunOrder("switch to wedge formation and move to 5, 4");
            }

            long allocBefore = GC.GetAllocatedBytesForCurrentThread();
            PollChannels();
            long allocAfter = GC.GetAllocatedBytesForCurrentThread();
            long delta = allocAfter - allocBefore;
            if (delta >= 0 && !hasPending) lastPollAllocBytes = delta;

            float dt = Time.deltaTime;
            UpdatePatrol(dt);
            UpdateSquadMovement(dt);
            UpdateVisualEffects(dt);
            UpdateCameraPosition(dt);
        }

        void PollChannels()
        {
            if (squadReady && squadHost != null)
            {
                while (squadChannel.TryReceive(out ChannelReply reply))
                {
                    if (hasPending && pendingMode == 0 && reply.Ticket == pendingTicket) hasPending = false;
                    HandleSquadReply(reply);
                }
            }

            if (campReady && campHost != null)
            {
                while (campChannel.TryReceive(out ChannelReply reply))
                {
                    if (hasPending && pendingMode == 1 && reply.Ticket == pendingTicket) hasPending = false;
                    HandleJsonToolReply(reply, null);
                }
            }

            if (voiceReady && voiceHost != null)
            {
                while (voiceChannel.TryReceiveDirect(out ChannelReply reply))
                {
                    if (hasPending && pendingMode == 2 && reply.Ticket == pendingTicket) hasPending = false;
                    HandleJsonToolReply(reply, null);
                }
                while (voiceChannel.TryReceive(out VoiceReply vReply))
                {
                    if (hasPending && pendingMode == 3 && vReply.Utterance == pendingTicket) hasPending = false;
                    string transcript = ReadText(ref vReply.Transcript, vReply.TranscriptLength);
                    lastTranscript = string.IsNullOrEmpty(transcript) ? "(silence)" : "\"" + transcript + "\"";
                    var chReply = new ChannelReply
                    {
                        Ticket = vReply.Ticket,
                        Reply = vReply.Reply,
                        Calls = vReply.Calls,
                        Reasoning = vReply.Reasoning,
                        Response = vReply.Response,
                    };
                    HandleJsonToolReply(chReply, transcript);
                }
            }
        }

        void HandleSquadReply(ChannelReply reply)
        {
            RecordReplyMetrics(in reply.Reply, ReadText(ref reply.Reasoning, reply.Reply.ReasoningLength));
            if (reply.Reply.Status != ReplyStatus.Called)
            {
                string resp = ReadText(ref reply.Response, reply.Reply.ResponseLength);
                lastCalls = reply.Reply.Status + (string.IsNullOrEmpty(resp) ? "" : ": " + resp);
                PushHistory("[" + reply.Reply.Status + "] " + inputOrder);
                return;
            }

            var callsBytes = Utf8.AsSpan(ref reply.Calls, reply.Reply.Utf8Length);
            var commands = MemoryMarshal.Cast<byte, SquadCommand>(callsBytes);
            var sb = new StringBuilder(128);
            for (int i = 0; i < commands.Length; i++)
            {
                if (i > 0) sb.Append("  |  ");
                sb.Append(ApplySquadCommand(in commands[i]));
            }
            lastCalls = sb.Length > 0 ? sb.ToString() : "(no calls)";
            PushHistory(FormatSeconds(reply.Reply) + " " + lastCalls);
        }

        string ApplySquadCommand(in SquadCommand command)
        {
            switch (command.Kind)
            {
                case SquadCommandKind.MoveTo:
                {
                    float rx = command.MoveTo.X;
                    float ry = command.MoveTo.Y;
                    float cx = Mathf.Clamp(rx, -ArenaHalfExtent, ArenaHalfExtent);
                    float cy = Mathf.Clamp(ry, -ArenaHalfExtent, ArenaHalfExtent);
                    squadDest = new Vector3(cx, 0f, cy);
                    squadMoving = true;
                    followingTarget = "";
                    engagingTarget = "";
                    waypointTimer = 6f;
                    if (WaypointMarker != null) WaypointMarker.position = new Vector3(cx, 0f, cy);
                    return string.Format(CultureInfo.InvariantCulture, "move_to({0:0.#}, {1:0.#})", rx, ry);
                }
                case SquadCommandKind.Attack:
                {
                    string target = ReadUtf8Text64(in command.Attack.Target);
                    engagingTarget = target;
                    followingTarget = "";
                    Vector3 targetPos = ResolveWorldTargetPos(target, OrcChieftain != null ? OrcChieftain.position : new Vector3(10f, 0f, 6f));
                    Vector3 dir = (squadPos - targetPos);
                    dir.y = 0f;
                    if (dir.sqrMagnitude < 0.01f) dir = Vector3.back;
                    dir.Normalize();
                    squadDest = targetPos + dir * 2.6f;
                    squadMoving = true;
                    attackPulse = 1.8f;
                    waypointTimer = 2f;
                    if (WaypointMarker != null) WaypointMarker.position = targetPos;
                    return "attack(\"" + target + "\")";
                }
                case SquadCommandKind.GiveItem:
                {
                    int idx = (int)command.GiveItem.Item;
                    int count = Math.Max(1, command.GiveItem.Count);
                    if (idx >= 0 && idx < inventory.Length) inventory[idx] += count;
                    string recipient = ReadUtf8Text32(in command.GiveItem.To);
                    if (string.IsNullOrEmpty(recipient)) recipient = "Lyra";
                    givePulse = 1.6f;
                    return "give_item(" + count + " " + command.GiveItem.Item + " -> " + recipient + ")";
                }
                case SquadCommandKind.SetFormation:
                {
                    currentFormation = (command.Presence & 0x1u) != 0 ? command.SetFormation.Formation : Formation.Line;
                    return "set_formation(" + currentFormation + ")";
                }
                case SquadCommandKind.Follow:
                {
                    string target = ReadUtf8Text32(in command.Follow.Target);
                    if (string.IsNullOrEmpty(target)) target = "Mira";
                    followingTarget = target;
                    engagingTarget = "";
                    squadMoving = false;
                    return "follow(\"" + target + "\")";
                }
                case SquadCommandKind.Stop:
                {
                    squadMoving = false;
                    followingTarget = "";
                    engagingTarget = "";
                    waypointTimer = 0f;
                    return "stop()";
                }
                default:
                    return command.Kind.ToString();
            }
        }

        void HandleJsonToolReply(ChannelReply reply, string spokenTranscript)
        {
            RecordReplyMetrics(in reply.Reply, ReadText(ref reply.Reasoning, reply.Reply.ReasoningLength));
            if (!reply.Reply.HasCalls)
            {
                string resp = ReadText(ref reply.Response, reply.Reply.ResponseLength);
                lastCalls = reply.Reply.Status + (string.IsNullOrEmpty(resp) ? "" : ": " + resp);
                string prefix = spokenTranscript != null ? "[Voice] \"" + spokenTranscript + "\" -> " : "";
                PushHistory(prefix + lastCalls);
                return;
            }

            string callsJson = ReadCallsText(ref reply.Calls, reply.Reply.Utf8Length);
            var parsed = JsonValue.Parse(callsJson);
            var sb = new StringBuilder(128);
            if (parsed.Kind == JsonKind.Array)
            {
                for (int i = 0; i < parsed.Items.Count; i++)
                {
                    var item = parsed.Items[i];
                    string name = item.GetString("name") ?? "";
                    var args = item.Get("arguments");
                    if (i > 0) sb.Append("  |  ");
                    sb.Append(DispatchJsonTool(name, args));
                }
            }
            lastCalls = sb.Length > 0 ? sb.ToString() : callsJson;
            string tag = spokenTranscript != null ? "[Voice \"" + spokenTranscript + "\"] " : "";
            PushHistory(FormatSeconds(reply.Reply) + " " + tag + lastCalls);
        }

        string DispatchJsonTool(string name, JsonValue args)
        {
            switch (name)
            {
                case "set_lights":
                {
                    lastRoom = args != null ? (args.GetString("room") ?? "camp") : "camp";
                    JsonValue onVal = args != null ? args.Get("on") : null;
                    if (onVal != null && onVal.Kind == JsonKind.Bool) campLightsOn = onVal.Boolean;
                    JsonValue brightVal = args != null ? args.Get("brightness") : null;
                    if (brightVal != null && int.TryParse(brightVal.Text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int b))
                        campBrightness = Mathf.Clamp(b, 0, 100);
                    ApplyCampVisuals();
                    return "set_lights(" + lastRoom + ", on=" + campLightsOn + ", brightness=" + campBrightness + ")";
                }
                case "lock_doors":
                {
                    JsonValue lockedVal = args != null ? args.Get("locked") : null;
                    if (lockedVal != null && lockedVal.Kind == JsonKind.Bool) campLocked = lockedVal.Boolean;
                    return "lock_doors(locked=" + campLocked + ")";
                }
                case "get_weather":
                {
                    string city = args != null ? (args.GetString("city") ?? "Unknown") : "Unknown";
                    weatherText = city + ": Clear Sky, 28C";
                    if (SunLight != null) SunLight.color = new Color(1.0f, 0.92f, 0.75f);
                    return "get_weather(\"" + city + "\")";
                }
                case "set_timer":
                {
                    int mins = 5;
                    JsonValue minVal = args != null ? args.Get("minutes") : null;
                    if (minVal != null && int.TryParse(minVal.Text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int m))
                        mins = Math.Max(1, m);
                    timerRemainingSeconds = mins * 60f;
                    timerLabel = mins + " min timer";
                    return "set_timer(" + mins + "m)";
                }
                case "play_music":
                {
                    string query = args != null ? (args.GetString("query") ?? "Music") : "Music";
                    musicNowPlaying = query;
                    musicPlaying = true;
                    return "play_music(\"" + query + "\")";
                }
                default:
                    return name + "()";
            }
        }

        void RecordReplyMetrics(in Reply r, string reasoning)
        {
            double prefillMs = r.PrefillTicks * 1000.0 / Reply.TicksPerSecond;
            double ttftMs = r.FirstTokenTicks * 1000.0 / Reply.TicksPerSecond;
            double totalMs = r.ElapsedTicks * 1000.0 / Reply.TicksPerSecond;
            double wallMs = (Time.realtimeSinceStartupAsDouble - queryStartTime) * 1000.0;
            double decodeMs = Math.Max(0.001, totalMs - prefillMs);
            double decodeTps = r.DecodeTokens > 0 ? r.DecodeTokens / (decodeMs / 1000.0) : 0.0;
            lastStatus = r.Status + " (" + r.Confidence.ToString("0.00", CultureInfo.InvariantCulture) + " conf)";
            lastReasoning = string.IsNullOrEmpty(reasoning) ? "—" : reasoning;
            lastTiming = string.Format(
                CultureInfo.InvariantCulture,
                "Engine {0:0.0} ms (Prefill {1:0.0} ms, {2:0} tok/s, {3} dec tok) | Wall {4:0.0} ms",
                totalMs, prefillMs, decodeTps, r.DecodeTokens, wallMs);
        }

        static string FormatSeconds(in Reply r)
        {
            double totalMs = r.ElapsedTicks * 1000.0 / Reply.TicksPerSecond;
            return totalMs.ToString("0.0", CultureInfo.InvariantCulture) + "ms";
        }

        void UpdatePatrol(float dt)
        {
            miraAngle += dt * 0.65f;
            miraPos = new Vector3(-9f + Mathf.Cos(miraAngle) * 3.8f, 0f, 5.5f + Mathf.Sin(miraAngle) * 3.8f);
            if (MiraPatrol != null)
            {
                Vector3 prev = MiraPatrol.position;
                MiraPatrol.position = miraPos;
                Vector3 look = miraPos - prev;
                look.y = 0f;
                if (look.sqrMagnitude > 0.0001f) MiraPatrol.rotation = Quaternion.Slerp(MiraPatrol.rotation, Quaternion.LookRotation(look), dt * 8f);
            }
        }

        void UpdateSquadMovement(float dt)
        {
            if (!string.IsNullOrEmpty(followingTarget))
            {
                Vector3 followPos = ResolveWorldTargetPos(followingTarget, miraPos);
                Vector3 toTarget = followPos - squadPos;
                toTarget.y = 0f;
                float dist = toTarget.magnitude;
                if (dist > 2.8f)
                {
                    Vector3 stepDir = toTarget / dist;
                    squadFacing = Vector3.Slerp(squadFacing, stepDir, dt * 8f);
                    squadPos += stepDir * (MoveSpeed * 0.9f * dt);
                }
            }
            else if (squadMoving)
            {
                Vector3 delta = squadDest - squadPos;
                delta.y = 0f;
                float dist = delta.magnitude;
                float step = MoveSpeed * dt;
                if (dist <= step)
                {
                    squadPos = squadDest;
                    squadMoving = false;
                }
                else
                {
                    Vector3 dir = delta / dist;
                    squadFacing = Vector3.Slerp(squadFacing, dir, dt * 10f);
                    squadPos += dir * step;
                }
            }

            if (!string.IsNullOrEmpty(engagingTarget) && !squadMoving)
            {
                Vector3 targetPos = ResolveWorldTargetPos(engagingTarget, OrcChieftain != null ? OrcChieftain.position : Vector3.zero);
                Vector3 look = targetPos - squadPos;
                look.y = 0f;
                if (look.sqrMagnitude > 0.01f) squadFacing = Vector3.Slerp(squadFacing, look.normalized, dt * 10f);
            }

            UpdateSquadTransforms(dt);
        }

        void UpdateSquadTransforms(float dt)
        {
            if (SquadRoot != null)
            {
                SquadRoot.position = squadPos;
                if (squadFacing.sqrMagnitude > 0.001f) SquadRoot.rotation = Quaternion.LookRotation(squadFacing, Vector3.up);
            }

            if (SquadMembers == null) return;
            bool isStriding = squadMoving || !string.IsNullOrEmpty(followingTarget);
            for (int i = 0; i < SquadMembers.Length; i++)
            {
                Transform unit = SquadMembers[i];
                if (unit == null) continue;
                Vector3 localSlot = GetFormationSlot(i, currentFormation);
                float bob = isStriding ? Mathf.Abs(Mathf.Sin(Time.time * 10f + i * 1.1f)) * 0.14f : 0f;
                if (i == 0 && attackPulse > 0f && !squadMoving)
                    localSlot.z += Mathf.Sin(attackPulse * 12f) * 0.45f;
                if (i == 1 && givePulse > 0f)
                    bob += Mathf.Abs(Mathf.Sin(givePulse * 8f)) * 0.35f;
                Vector3 desiredWorld = squadPos + (squadFacing.sqrMagnitude > 0.001f ? Quaternion.LookRotation(squadFacing) * localSlot : localSlot);
                desiredWorld.y = bob;
                unit.position = Vector3.Lerp(unit.position, desiredWorld, Mathf.Clamp01(dt * 9f));
                if (squadFacing.sqrMagnitude > 0.001f)
                    unit.rotation = Quaternion.Slerp(unit.rotation, Quaternion.LookRotation(squadFacing, Vector3.up), Mathf.Clamp01(dt * 9f));
            }
        }

        static Vector3 GetFormationSlot(int index, Formation formation)
        {
            switch (formation)
            {
                case Formation.Line:
                    return new Vector3((index - 1.5f) * 1.8f, 0f, 0f);
                case Formation.Wedge:
                    if (index == 0) return new Vector3(0f, 0f, 1.3f);
                    if (index == 1) return new Vector3(-1.4f, 0f, -0.3f);
                    if (index == 2) return new Vector3(1.4f, 0f, -0.3f);
                    return new Vector3(0f, 0f, -1.6f);
                case Formation.Circle:
                {
                    float a = index * Mathf.PI * 0.5f;
                    return new Vector3(Mathf.Sin(a) * 1.6f, 0f, Mathf.Cos(a) * 1.6f);
                }
                case Formation.Column:
                    return new Vector3(0f, 0f, (1.5f - index) * 1.6f);
                default:
                    return Vector3.zero;
            }
        }

        void UpdateVisualEffects(float dt)
        {
            if (waypointTimer > 0f) waypointTimer = Mathf.Max(0f, waypointTimer - dt);
            if (WaypointMarker != null)
            {
                bool show = waypointTimer > 0f && (squadMoving || attackPulse > 0f);
                if (WaypointMarker.gameObject.activeSelf != show) WaypointMarker.gameObject.SetActive(show);
                if (show && WaypointRing != null)
                {
                    float s = 1.0f + Mathf.Sin(Time.time * 6f) * 0.18f;
                    WaypointRing.localScale = new Vector3(s, 0.05f, s);
                }
            }

            if (attackPulse > 0f)
            {
                attackPulse = Mathf.Max(0f, attackPulse - dt);
                if (OrcBodyRenderer != null)
                    OrcBodyRenderer.material.color = Color.Lerp(new Color(0.82f, 0.22f, 0.24f), new Color(1f, 0.85f, 0.2f), Mathf.PingPong(attackPulse * 6f, 1f));
            }
            else if (OrcBodyRenderer != null)
            {
                OrcBodyRenderer.material.color = new Color(0.82f, 0.22f, 0.24f);
            }

            if (givePulse > 0f) givePulse = Mathf.Max(0f, givePulse - dt);

            float targetGate = campLocked ? 1f : 0f;
            gateClosedAmount = Mathf.MoveTowards(gateClosedAmount, targetGate, dt * 4f);
            if (CampGate != null)
            {
                Vector3 p = CampGate.localPosition;
                p.y = Mathf.Lerp(0.12f, 1.35f, gateClosedAmount);
                CampGate.localPosition = p;
                var ren = CampGate.GetComponent<Renderer>();
                if (ren != null)
                    ren.material.color = campLocked ? new Color(0.95f, 0.25f, 0.25f) : new Color(0.2f, 0.9f, 0.5f);
            }

            if (timerRemainingSeconds > 0f)
                timerRemainingSeconds = Mathf.Max(0f, timerRemainingSeconds - dt);

            if (musicPlaying)
            {
                musicPulsePhase += dt * 4f;
                Color pulse = Color.HSVToRGB((musicPulsePhase * 0.12f) % 1f, 0.75f, 1f);
                if (MusicOrbLight != null)
                {
                    MusicOrbLight.enabled = true;
                    MusicOrbLight.color = pulse;
                    MusicOrbLight.intensity = 18f + Mathf.Sin(musicPulsePhase * 2f) * 6f;
                }
                if (MusicOrbRenderer != null) MusicOrbRenderer.material.color = pulse;
                if (MusicOrb != null)
                    MusicOrb.localPosition = new Vector3(0f, 1.6f + Mathf.Sin(musicPulsePhase) * 0.22f, 0f);
            }
            else
            {
                if (MusicOrbLight != null) MusicOrbLight.enabled = false;
                if (MusicOrbRenderer != null) MusicOrbRenderer.material.color = new Color(0.28f, 0.30f, 0.42f);
            }
        }

        void ApplyCampVisuals()
        {
            float level = campLightsOn ? Mathf.Clamp01(campBrightness / 100f) : 0f;
            if (CampLights != null)
            {
                for (int i = 0; i < CampLights.Length; i++)
                {
                    if (CampLights[i] == null) continue;
                    CampLights[i].enabled = campLightsOn && campBrightness > 0;
                    CampLights[i].intensity = 28f * level;
                }
            }
            if (CampLampRenderers != null)
            {
                Color c = Color.Lerp(new Color(0.15f, 0.15f, 0.18f), new Color(1f, 0.8f, 0.35f), level);
                for (int i = 0; i < CampLampRenderers.Length; i++)
                    if (CampLampRenderers[i] != null) CampLampRenderers[i].material.color = c;
            }
        }

        void UpdateCameraPosition(float dt)
        {
            if (DemoCamera == null) return;
            Vector3 desiredFocus = followCamera ? squadPos + new Vector3(0f, 0.8f, 0f) : new Vector3(0f, 0.5f, 0.8f);
            camFocus = Vector3.Lerp(camFocus, desiredFocus, dt * 5f);
            Quaternion rot = Quaternion.Euler(camPitch, camYaw, 0f);
            DemoCamera.transform.position = camFocus - (rot * Vector3.forward) * camDistance;
            DemoCamera.transform.LookAt(camFocus);
        }

        Vector3 ResolveWorldTargetPos(string target, Vector3 fallback)
        {
            string lower = (target ?? "").ToLowerInvariant();
            if (lower.Contains("mira") && MiraPatrol != null) return MiraPatrol.position;
            if (lower.Contains("tower") || lower.Contains("watch"))
                return Watchtower != null ? Watchtower.position : new Vector3(12f, 0f, 9f);
            if (lower.Contains("orc") || lower.Contains("chief") || lower.Contains("enemy"))
                return OrcChieftain != null ? OrcChieftain.position : new Vector3(10f, 0f, 6f);
            if (lower.Contains("camp") || lower.Contains("gate"))
                return new Vector3(-10f, 0f, -7f);
            if (lower.Contains("lyra") && SquadMembers != null && SquadMembers.Length > 1 && SquadMembers[1] != null)
                return SquadMembers[1].position;
            return fallback;
        }

        string ReadCallsText(ref CallsUtf8 calls, int len)
        {
            int clamped = Math.Max(0, Math.Min(len, decodeBuffer.Length));
            if (clamped == 0) return "";
            Utf8.AsSpan(ref calls, clamped).CopyTo(decodeBuffer);
            return Encoding.UTF8.GetString(decodeBuffer, 0, clamped);
        }

        string ReadText(ref TextUtf8 text, int len)
        {
            int clamped = Math.Max(0, Math.Min(len, decodeBuffer.Length));
            if (clamped == 0) return "";
            Utf8.AsSpan(ref text, clamped).CopyTo(decodeBuffer);
            return Encoding.UTF8.GetString(decodeBuffer, 0, clamped);
        }

        static string ReadUtf8Text64(in Utf8Text64 text)
        {
            var span = text.Span;
            if (span.Length == 0) return "";
            return Encoding.UTF8.GetString(span.ToArray());
        }

        static string ReadUtf8Text32(in Utf8Text32 text)
        {
            var span = text.Span;
            if (span.Length == 0) return "";
            return Encoding.UTF8.GetString(span.ToArray());
        }

        void PushHistory(string entry)
        {
            history.Insert(0, entry);
            if (history.Count > 18) history.RemoveAt(history.Count - 1);
        }

        void EnsureStyles()
        {
            if (stylesReady) return;
            stylesReady = true;

            darkBgTex = MakeSolid(new Color(0.06f, 0.08f, 0.12f, 0.94f));
            panelTex = MakeSolid(new Color(0.08f, 0.11f, 0.17f, 0.84f));
            cardTex = MakeSolid(new Color(0.12f, 0.16f, 0.24f, 0.90f));
            inputTex = MakeSolid(new Color(0.04f, 0.06f, 0.09f, 0.98f));
            accentTex = MakeSolid(new Color(0.14f, 0.56f, 0.88f, 1f));
            accentHoverTex = MakeSolid(new Color(0.20f, 0.66f, 0.96f, 1f));
            buttonTex = MakeSolid(new Color(0.16f, 0.21f, 0.30f, 0.96f));
            buttonHoverTex = MakeSolid(new Color(0.23f, 0.30f, 0.42f, 0.98f));
            greenPillTex = MakeSolid(new Color(0.10f, 0.48f, 0.32f, 0.96f));
            amberPillTex = MakeSolid(new Color(0.58f, 0.38f, 0.10f, 0.96f));

            panelStyle = new GUIStyle(GUI.skin.box);
            panelStyle.normal.background = panelTex;
            panelStyle.padding = new RectOffset(10, 10, 10, 10);

            cardStyle = new GUIStyle(GUI.skin.box);
            cardStyle.normal.background = cardTex;
            cardStyle.padding = new RectOffset(8, 8, 8, 8);

            titleStyle = new GUIStyle(GUI.skin.label);
            titleStyle.fontSize = 15;
            titleStyle.fontStyle = FontStyle.Bold;
            titleStyle.normal.textColor = Color.white;

            labelStyle = new GUIStyle(GUI.skin.label);
            labelStyle.fontSize = 13;
            labelStyle.wordWrap = true;
            labelStyle.normal.textColor = new Color(0.92f, 0.95f, 0.99f);

            mutedStyle = new GUIStyle(GUI.skin.label);
            mutedStyle.fontSize = 12;
            mutedStyle.wordWrap = true;
            mutedStyle.normal.textColor = new Color(0.66f, 0.74f, 0.85f);

            monoStyle = new GUIStyle(GUI.skin.label);
            monoStyle.fontSize = 12;
            monoStyle.wordWrap = true;
            monoStyle.normal.textColor = new Color(0.52f, 0.92f, 0.78f);

            inputStyle = new GUIStyle(GUI.skin.textField);
            inputStyle.fontSize = 13;
            inputStyle.clipping = TextClipping.Clip;
            inputStyle.padding = new RectOffset(8, 8, 6, 6);
            inputStyle.normal.background = inputTex;
            inputStyle.focused.background = inputTex;
            inputStyle.normal.textColor = Color.white;
            inputStyle.focused.textColor = Color.white;

            buttonStyle = new GUIStyle(GUI.skin.button);
            buttonStyle.fontSize = 12;
            buttonStyle.fontStyle = FontStyle.Bold;
            buttonStyle.alignment = TextAnchor.MiddleCenter;
            buttonStyle.padding = new RectOffset(8, 8, 6, 6);
            buttonStyle.normal.background = buttonTex;
            buttonStyle.hover.background = buttonHoverTex;
            buttonStyle.active.background = accentTex;
            buttonStyle.normal.textColor = new Color(0.94f, 0.96f, 1f);
            buttonStyle.hover.textColor = Color.white;

            activeButtonStyle = new GUIStyle(buttonStyle);
            activeButtonStyle.normal.background = accentTex;
            activeButtonStyle.hover.background = accentHoverTex;
            activeButtonStyle.normal.textColor = Color.white;

            presetButtonStyle = new GUIStyle(buttonStyle);
            presetButtonStyle.wordWrap = true;
            presetButtonStyle.alignment = TextAnchor.MiddleLeft;
            presetButtonStyle.padding = new RectOffset(10, 10, 7, 7);

            nameplateStyle = new GUIStyle(GUI.skin.box);
            nameplateStyle.fontSize = 11;
            nameplateStyle.fontStyle = FontStyle.Bold;
            nameplateStyle.alignment = TextAnchor.MiddleCenter;
            nameplateStyle.padding = new RectOffset(6, 6, 2, 2);
            nameplateStyle.normal.background = darkBgTex;
            nameplateStyle.normal.textColor = Color.white;
        }

        static Texture2D MakeSolid(Color color)
        {
            var tex = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            tex.SetPixel(0, 0, color);
            tex.Apply();
            return tex;
        }

        void OnGUI()
        {
            EnsureStyles();
            HandleCameraDragAndScroll();

            float uiScale = Mathf.Clamp(Mathf.Min(Screen.width / 1400f, Screen.height / 820f), 0.35f, 1.35f);
            float vw = Screen.width / uiScale;
            float vh = Screen.height / uiScale;
            Matrix4x4 prevMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.Scale(new Vector3(uiScale, uiScale, 1f));

            DrawWorldNameplates(uiScale, vw, vh);
            DrawTopBar(vw);
            DrawLeftDock(vw, vh);
            DrawRightTelemetry(vw, vh);
            DrawBottomCameraStrip(vw, vh);

            GUI.matrix = prevMatrix;
        }

        void HandleCameraDragAndScroll()
        {
            Event ev = Event.current;
            if (ev.type == EventType.MouseDown && ev.button == 1)
            {
                draggingCamera = true;
                lastMousePos = ev.mousePosition;
                ev.Use();
            }
            else if (ev.type == EventType.MouseUp && ev.button == 1)
            {
                draggingCamera = false;
            }
            else if (ev.type == EventType.MouseDrag && draggingCamera)
            {
                Vector2 d = ev.mousePosition - lastMousePos;
                lastMousePos = ev.mousePosition;
                camYaw += d.x * 0.35f;
                camPitch = Mathf.Clamp(camPitch + d.y * 0.25f, 18f, 75f);
                ev.Use();
            }
            else if (ev.type == EventType.ScrollWheel)
            {
                camDistance = Mathf.Clamp(camDistance + ev.delta.y * 1.2f, 12f, 64f);
                ev.Use();
            }
        }

        void DrawWorldNameplates(float uiScale, float vw, float vh)
        {
            if (DemoCamera == null) return;
            float sideMargin = Mathf.Clamp(vw * 0.215f, 285f, 325f) + 12f;
            DrawNameplate(squadPos + new Vector3(0f, 3.0f, 0f), "Squad [" + currentFormation + "]", uiScale, vw, vh, sideMargin);
            if (SquadMembers != null && SquadMembers.Length > 1 && SquadMembers[1] != null)
                DrawNameplate(SquadMembers[1].position + new Vector3(-0.3f, 1.7f, -0.3f), "Lyra (" + inventory[0] + " Potion)", uiScale, vw, vh, sideMargin);
            if (MiraPatrol != null)
                DrawNameplate(MiraPatrol.position + new Vector3(0f, 2.0f, 0f), "Mira (Patrol)", uiScale, vw, vh, sideMargin);
            if (OrcChieftain != null)
                DrawNameplate(OrcChieftain.position + new Vector3(0f, 2.6f, 0f), "Orc Chieftain", uiScale, vw, vh, sideMargin);
            if (Watchtower != null)
                DrawNameplate(Watchtower.position + new Vector3(0f, 4.3f, 0f), "Watchtower", uiScale, vw, vh, sideMargin);
            if (CampGate != null)
                DrawNameplate(new Vector3(-10f, 2.9f, -7f), "Camp Outpost (" + (campLocked ? "LOCKED" : "OPEN") + ")", uiScale, vw, vh, sideMargin);
            if (MusicOrb != null && musicPlaying)
                DrawNameplate(MusicOrb.position + new Vector3(0f, 1.3f, 0f), "Playing: " + musicNowPlaying, uiScale, vw, vh, sideMargin);
        }

        void DrawNameplate(Vector3 worldPos, string text, float uiScale, float vw, float vh, float sideMargin)
        {
            Vector3 sp = DemoCamera.WorldToScreenPoint(worldPos);
            if (sp.z <= 0f) return;
            Vector2 size = nameplateStyle.CalcSize(new GUIContent(text));
            float x = (sp.x / uiScale) - size.x * 0.5f;
            float y = ((Screen.height - sp.y) / uiScale) - size.y;
            if (x < sideMargin || x + size.x > vw - sideMargin || y < 56f || y > vh - 48f) return;
            GUI.Box(new Rect(x, y, size.x + 6f, size.y), text, nameplateStyle);
        }

        void DrawTopBar(float vw)
        {
            GUI.Box(new Rect(8f, 8f, vw - 16f, 44f), GUIContent.none, panelStyle);
            GUILayout.BeginArea(new Rect(16f, 14f, vw - 32f, 34f));
            GUILayout.BeginHorizontal();

            GUILayout.Label("NEEDLE 3 + WHISTLE DEMO", titleStyle, GUILayout.Width(250f));

            if (GUILayout.Button("1. Squad (6 Tools)", activeMode == 0 ? activeButtonStyle : buttonStyle, GUILayout.Width(135f))) SelectMode(0);
            if (GUILayout.Button("2. Smart Camp (2)", activeMode == 1 ? activeButtonStyle : buttonStyle, GUILayout.Width(135f))) SelectMode(1);
            if (GUILayout.Button("3. Assistant + Voice", activeMode == 2 ? activeButtonStyle : buttonStyle, GUILayout.Width(155f))) SelectMode(2);

            GUILayout.Space(12f);
            GUILayout.Label("Threads:", mutedStyle, GUILayout.Width(58f));
            if (GUILayout.Button("2T", selectedThreads == 2 ? activeButtonStyle : buttonStyle, GUILayout.Width(36f))) SetThreads(2);
            if (GUILayout.Button("4T", selectedThreads == 4 ? activeButtonStyle : buttonStyle, GUILayout.Width(36f))) SetThreads(4);
            if (GUILayout.Button("8T", selectedThreads == 8 ? activeButtonStyle : buttonStyle, GUILayout.Width(36f))) SetThreads(8);

            GUILayout.FlexibleSpace();
            string provider = Kernels.Provider ?? "uninitialized";
            GUILayout.Label("Backend: " + provider + "  |  Hot-Path Poll: " + lastPollAllocBytes + " B", monoStyle);
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        void DrawLeftDock(float vw, float vh)
        {
            float w = Mathf.Clamp(vw * 0.215f, 285f, 325f);
            float h = vh - 110f;
            Rect dock = new Rect(8f, 60f, w, h);
            GUI.Box(dock, GUIContent.none, panelStyle);

            GUILayout.BeginArea(new Rect(dock.x + 10f, dock.y + 10f, dock.width - 20f, dock.height - 20f));

            bool modeReady = activeMode == 0 ? squadReady : activeMode == 1 ? campReady : voiceReady;
            string pillText = hasPending ? "THINKING..." : modeReady ? "ENGINE READY (" + activeThreadId + "T)" : "LOADING MODEL...";
            GUIStyle pill = new GUIStyle(GUI.skin.box);
            pill.fontStyle = FontStyle.Bold;
            pill.fontSize = 12;
            pill.alignment = TextAnchor.MiddleCenter;
            pill.normal.textColor = Color.white;
            pill.normal.background = modeReady && !hasPending ? greenPillTex : amberPillTex;
            GUILayout.Box(pillText, pill, GUILayout.ExpandWidth(true), GUILayout.Height(24f));

            if (!string.IsNullOrEmpty(initError))
            {
                GUILayout.Label("Error: " + initError, labelStyle);
            }

            GUILayout.Space(6f);
            GUILayout.Label("Type Natural-Language Command (Press Enter):", mutedStyle);

            Event ev = Event.current;
            bool submitOnEnter = ev.type == EventType.KeyDown
                && (ev.keyCode == KeyCode.Return || ev.keyCode == KeyCode.KeypadEnter)
                && GUI.GetNameOfFocusedControl() == InputControlName;
            if (submitOnEnter)
            {
                ev.Use();
                RunOrder(inputOrder);
            }

            GUILayout.BeginHorizontal();
            GUI.SetNextControlName(InputControlName);
            inputOrder = GUILayout.TextField(inputOrder ?? "", 256, inputStyle, GUILayout.Height(30f), GUILayout.ExpandWidth(true), GUILayout.MinWidth(100f));
            if (GUILayout.Button("Send", activeButtonStyle, GUILayout.Width(58f), GUILayout.Height(30f)))
                RunOrder(inputOrder);
            GUILayout.EndHorizontal();

            GUILayout.Space(10f);
            GUILayout.Label("One-Click Demo Prompts:", titleStyle);
            presetScroll = GUILayout.BeginScrollView(presetScroll);

            if (activeMode == 0)
            {
                PresetButton("move to 10, 8", "MoveTo(10, 8)");
                PresetButton("switch to wedge formation", "SetFormation(Wedge)");
                PresetButton("attack the orc chieftain", "Attack(\"orc chieftain\")");
                PresetButton("give 3 potions to Lyra", "GiveItem(Potion, 3, \"Lyra\")");
                PresetButton("follow Mira", "Follow(\"Mira\")");
                PresetButton("switch to circle formation and follow Mira", "Compound Call (2 tools)");
                PresetButton("move to -8, -5", "MoveTo(-8, -5)");
                PresetButton("attack the watchtower", "Attack(\"watchtower\")");
                PresetButton("give 50 gold to Lyra", "GiveItem(Gold, 50, \"Lyra\")");
                PresetButton("stop everything", "Stop()");
            }
            else if (activeMode == 1)
            {
                PresetButton("turn off the camp lights", "set_lights(off)");
                PresetButton("dim the camp lights to 25 and unlock the gate", "Compound (lights + gate)");
                PresetButton("turn camp lights on bright and lock the gate", "Compound (100% + lock)");
                PresetButton("lock the camp doors", "lock_doors(true)");
                PresetButton("unlock the gate and set lights to 80", "Compound (open + 80%)");
            }
            else
            {
                PresetButton("what's it like in Lagos right now?", "get_weather(\"Lagos\")");
                PresetButton("set a timer for 12 minutes and play some Miles Davis", "Compound (timer + music)");
                PresetButton("play Daft Punk", "play_music(\"Daft Punk\")");
                PresetButton("set a 5 minute timer", "set_timer(5)");

                GUILayout.Space(10f);
                GUILayout.Label("Whistle Audio Clips -> Speech-to-Tool:", titleStyle);
                VoiceClipButton("record.wav", "Speak record.wav (4.5s)", "Whistle + Needle (~12-34 ms)");
                VoiceClipButton("hotword.wav", "Speak hotword.wav (17.6s)", "Whistle STT (~40 ms)");
                VoiceClipButton("test.wav", "Speak test.wav (20s)", "Whistle STT (~42 ms)");

                GUILayout.Space(6f);
                audioEnabled = GUILayout.Toggle(audioEnabled, " Play Audio Clip Aloud");
            }

            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        void PresetButton(string prompt, string hint)
        {
            if (GUILayout.Button(prompt + "  [" + hint + "]", presetButtonStyle, GUILayout.MinHeight(32f)))
                RunOrder(prompt);
            GUILayout.Space(4f);
        }

        void VoiceClipButton(string fileName, string title, string hint)
        {
            if (GUILayout.Button(title + "  [" + hint + "]", activeButtonStyle, GUILayout.MinHeight(34f)))
                RunVoiceClip(fileName);
            GUILayout.Space(4f);
        }

        void DrawRightTelemetry(float vw, float vh)
        {
            float w = Mathf.Clamp(vw * 0.215f, 285f, 325f);
            float h = vh - 110f;
            Rect panel = new Rect(vw - w - 8f, 60f, w, h);
            GUI.Box(panel, GUIContent.none, panelStyle);

            GUILayout.BeginArea(new Rect(panel.x + 10f, panel.y + 10f, panel.width - 20f, panel.height - 20f));

            GUILayout.BeginVertical(cardStyle);
            GUILayout.Label("Latest Tool Call & Speed", titleStyle);
            GUILayout.Label("Status: " + lastStatus, monoStyle);
            GUILayout.Label("Calls: " + lastCalls, labelStyle);
            if (activeMode == 2) GUILayout.Label("Transcript: " + lastTranscript, mutedStyle);
            GUILayout.Label("Timing: " + lastTiming, monoStyle);
            GUILayout.Label("Reasoning: " + lastReasoning, mutedStyle);
            GUILayout.EndVertical();

            GUILayout.Space(8f);
            GUILayout.BeginVertical(cardStyle);
            GUILayout.Label("Live 3D Scene State", titleStyle);
            GUILayout.Label(string.Format(CultureInfo.InvariantCulture, "Squad Pos: ({0:0.0}, {1:0.0})  |  Formation: {2}", squadPos.x, squadPos.z, currentFormation), labelStyle);
            GUILayout.Label("Action: " + (!string.IsNullOrEmpty(followingTarget) ? "Following " + followingTarget : !string.IsNullOrEmpty(engagingTarget) ? "Attacking " + engagingTarget : squadMoving ? "Moving" : "Idle"), mutedStyle);
            GUILayout.Label(string.Format(CultureInfo.InvariantCulture, "Lyra Bag: {0} Potion, {1} Sword, {2} Shield, {3} Arrow, {4} Gold", inventory[0], inventory[1], inventory[2], inventory[3], inventory[4]), mutedStyle);
            GUILayout.Label("Camp Lights: " + (campLightsOn ? "ON (" + campBrightness + "%)" : "OFF") + "  |  Gate: " + (campLocked ? "LOCKED" : "OPEN"), mutedStyle);
            GUILayout.Label("Weather: " + weatherText + "  |  Music: " + musicNowPlaying, mutedStyle);
            if (timerRemainingSeconds > 0f)
            {
                int secs = Mathf.CeilToInt(timerRemainingSeconds);
                GUILayout.Label(string.Format(CultureInfo.InvariantCulture, "Timer: {0} ({1:00}:{2:00})", timerLabel, secs / 60, secs % 60), monoStyle);
            }
            GUILayout.EndVertical();

            GUILayout.Space(8f);
            GUILayout.Label("Execution Log:", titleStyle);
            historyScroll = GUILayout.BeginScrollView(historyScroll);
            for (int i = 0; i < history.Count; i++)
                GUILayout.Label(history[i], mutedStyle);
            GUILayout.EndScrollView();

            GUILayout.EndArea();
        }

        void DrawBottomCameraStrip(float vw, float vh)
        {
            float stripW = 505f;
            float x = (vw - stripW) * 0.5f;
            float y = vh - 42f;
            GUI.Box(new Rect(x, y, stripW, 34f), GUIContent.none, panelStyle);
            GUILayout.BeginArea(new Rect(x + 8f, y + 4f, stripW - 16f, 26f));
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Reset Arena", buttonStyle, GUILayout.Width(96f))) ResetArenaState();
            if (GUILayout.Button(followCamera ? "Cam: Follow" : "Cam: Arena", followCamera ? activeButtonStyle : buttonStyle, GUILayout.Width(96f)))
                followCamera = !followCamera;
            if (GUILayout.Button("Reset View", buttonStyle, GUILayout.Width(88f)))
            {
                camYaw = 0f;
                camPitch = 44f;
                camDistance = 43f;
                camFocus = new Vector3(0f, 0.5f, 0.8f);
            }
            if (GUILayout.Button("-", buttonStyle, GUILayout.Width(32f))) camDistance = Mathf.Clamp(camDistance + 3f, 12f, 64f);
            if (GUILayout.Button("+", buttonStyle, GUILayout.Width(32f))) camDistance = Mathf.Clamp(camDistance - 3f, 12f, 64f);
            GUILayout.Space(8f);
            GUILayout.Label("Right-Drag Orbit", mutedStyle, GUILayout.Width(118f));
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }
    }
}
