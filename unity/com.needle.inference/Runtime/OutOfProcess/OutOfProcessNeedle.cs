using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Needle
{
    /// <summary>
    /// N70: Unity client for `needle serve` (NativeAOT out-of-process inference).
    ///
    /// For runtimes with nothing fast: instead of loading Needle.Inference.dll
    /// in-process, this spawns the `needle` binary as a child, talks to it over
    /// stdin/stdout (one JSON request per line, one reply per line), and kills
    /// the child when the GameObject disables, the application quits, or the
    /// Unity process exits. Desktop players and the Editor only: WebGL,
    /// Android and iOS cannot spawn processes, so this component logs an error
    /// and disables itself there.
    ///
    /// Protocol (mirrors src/Needle.Cli/Serve.cs):
    ///   send {"id":7,"type":"query","text":"..."}      (or "tool_result",
    ///        "reset", "ping")
    ///   recv {"id":7,"success":true,"error":null,"type":"call",
    ///         "function_calls":[...], ... "total_time_ms":91.2}
    /// Overhead check: client wall-clock round trip minus the reply's
    /// total_time_ms is the pipe cost; acceptance is &lt;= 1 ms on top of
    /// in-process net10 latency for a warm query (ping measures pure pipe).
    ///
    /// Usage: put on any GameObject, set Executable/ModelFile/Tools, call
    /// Ask() from UI, read replies in Update() via the Replied event or by
    /// polling Drain(). Replies arrive on the Unity thread (Update pumps a
    /// lock-guarded queue fed by a background stdout reader), so handlers may
    /// touch the scene directly.
    /// </summary>
    public sealed class OutOfProcessNeedle : MonoBehaviour
    {
        [Header("Server (desktop players and Editor only)")]
        [Tooltip("Needle binary to spawn. Bare name resolves via PATH; absolute path also works.")]
        public string Executable = "needle";

        [Tooltip("Model file. Relative names resolve under Application.streamingAssetsPath.")]
        public string ModelFile = "needle3.cact";

        [Tooltip("Tool schema. Takes precedence over ToolsFile when set.")]
        public TextAsset Tools;

        [Tooltip("Tool schema file. Relative names resolve under Application.streamingAssetsPath. Ignored when Tools is set.")]
        public string ToolsFile = string.Empty;

        [Header("Session")]
        [Tooltip("Inference threads inside the child. Mirrors `needle serve --threads`.")]
        public int Threads = 2;

        [Tooltip("Pass --stateless so every query starts a fresh conversation.")]
        public bool Stateless = true;

        [Tooltip("Extra `needle serve` flags, e.g. \"--no-date --max-reasoning 16\".")]
        public string ExtraArgs = string.Empty;

        /// <summary>Raised on the Unity thread for every server reply.</summary>
        public event Action<OutOfProcessReply> Replied;

        /// <summary>True after the child spawned and until it exits or fails.</summary>
        public bool Connected { get; private set; }

        /// <summary>Requests sent minus replies received.</summary>
        public int PendingRequests
        {
            get { lock (_gate) { return _sent - _received; } }
        }

        /// <summary>False on WebGL/Android/iOS, where processes cannot spawn.</summary>
        public static bool IsSupportedPlatform()
        {
            var platform = Application.platform;
            return platform == RuntimePlatform.WindowsEditor
                || platform == RuntimePlatform.OSXEditor
                || platform == RuntimePlatform.LinuxEditor
                || platform == RuntimePlatform.WindowsPlayer
                || platform == RuntimePlatform.OSXPlayer
                || platform == RuntimePlatform.LinuxPlayer;
        }

        /// <summary>Submit a query. Returns the request id (&gt;0), or -1 when not connected.</summary>
        public int Ask(string query)
        {
            return Send("query", query);
        }

        /// <summary>Feed a tool result back into the turn it answers. Returns the id, or -1.</summary>
        public int SendToolResult(string resultJson)
        {
            return Send("tool_result", resultJson);
        }

        /// <summary>Ask the child to forget the conversation. Returns the id, or -1.</summary>
        public int ResetServer()
        {
            return Send("reset", string.Empty);
        }

        /// <summary>No inference; measures pure pipe cost. Returns the id, or -1.</summary>
        public int Ping()
        {
            return Send("ping", string.Empty);
        }

        int Send(string type, string text)
        {
            System.Diagnostics.Process proc;
            lock (_gate)
            {
                if (!Connected || _proc == null || _nextId <= 0) return -1;
                proc = _proc;
            }
            var id = Interlocked.Increment(ref _nextId) - 1;
            var line = EncodeRequest(id, type, text == null ? string.Empty : text);
            try
            {
                lock (_writeGate)
                {
                    proc.StandardInput.WriteLine(line);
                    proc.StandardInput.Flush();
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("Needle server write failed: " + e.Message);
                return -1;
            }
            lock (_gate) { _sent++; }
            return id;
        }

        void Update()
        {
            if (!Connected) return;
            for (;;)
            {
                string line = null;
                lock (_gate)
                {
                    if (_replies.Count > 0) line = _replies.Dequeue();
                }
                if (line == null) break;
                Dispatch(line);
            }
            if (Volatile.Read(ref _readerDone) != 0)
            {
                lock (_gate)
                {
                    if (_replies.Count == 0 && Connected)
                    {
                        Connected = false;
                        Debug.LogError("Needle server exited; out-of-process inference is offline.");
                    }
                }
            }
        }

        void OnEnable()
        {
            _shutdown = 0;
            if (!IsSupportedPlatform())
            {
                Debug.LogError("OutOfProcessNeedle needs a desktop player or the Editor (this platform cannot spawn processes).");
                enabled = false;
                return;
            }
            string modelPath = ResolvePath(ModelFile);
            if (!System.IO.File.Exists(modelPath))
            {
                Debug.LogError("Needle model missing: " + modelPath);
                enabled = false;
                return;
            }
            string toolsPath;
            try
            {
                toolsPath = ResolveTools();
            }
            catch (Exception e)
            {
                Debug.LogError("Needle tools missing: " + e.Message);
                enabled = false;
                return;
            }
            var args = new StringBuilder("serve ");
            args.Append(QuoteArg(modelPath)).Append(' ').Append(QuoteArg(toolsPath));
            args.Append(" --threads ").Append(Math.Max(1, Threads));
            if (Stateless) args.Append(" --stateless");
            if (!string.IsNullOrEmpty(ExtraArgs)) args.Append(' ').Append(ExtraArgs);

            var proc = new System.Diagnostics.Process();
            proc.StartInfo.FileName = Executable;
            proc.StartInfo.Arguments = args.ToString();
            proc.StartInfo.UseShellExecute = false;
            proc.StartInfo.RedirectStandardInput = true;
            proc.StartInfo.RedirectStandardOutput = true;
            proc.StartInfo.RedirectStandardError = true;
            proc.StartInfo.CreateNoWindow = true;
            proc.StartInfo.StandardOutputEncoding = Encoding.UTF8;
            proc.StartInfo.StandardInputEncoding = Encoding.UTF8;
            try
            {
                if (!proc.Start())
                {
                    Debug.LogError("Could not start Needle server: " + Executable);
                    proc.Dispose();
                    enabled = false;
                    return;
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Could not start Needle server '" + Executable + "': " + e.Message);
                proc.Dispose();
                enabled = false;
                return;
            }
            lock (_gate)
            {
                _proc = proc;
                _replies.Clear();
                _sent = 0;
                _received = 0;
                _nextId = 1;
            }
            Volatile.Write(ref _readerDone, 0);
            Connected = true;
            var reader = new Thread(ReaderLoop) { IsBackground = true, Name = "needle-serve-stdout" };
            reader.Start(proc);
            var errors = new Thread(ErrorLoop) { IsBackground = true, Name = "needle-serve-stderr" };
            errors.Start(proc);
        }

        void OnDisable()
        {
            Shutdown();
        }

        void OnApplicationQuit()
        {
            Shutdown();
        }

        void Shutdown()
        {
            if (Interlocked.Exchange(ref _shutdown, 1) != 0) return;
            Connected = false;
            System.Diagnostics.Process proc;
            lock (_gate) { proc = _proc; _proc = null; }
            if (proc == null) return;
            try
            {
                try { proc.StandardInput.Close(); }
                catch (Exception) { }
                if (!proc.WaitForExit(500))
                {
                    try { proc.Kill(); }
                    catch (Exception) { }
                    proc.WaitForExit(2000);
                }
            }
            catch (Exception) { }
            finally
            {
                try { proc.Dispose(); }
                catch (Exception) { }
            }
        }

        void ReaderLoop(object state)
        {
            var proc = (System.Diagnostics.Process)state;
            try
            {
                for (;;)
                {
                    string line = proc.StandardOutput.ReadLine();
                    if (line == null) break;
                    if (line.Length == 0) continue;
                    lock (_gate)
                    {
                        if (_replies.Count < Math.Max(64, MaxQueued)) _replies.Enqueue(line);
                    }
                }
            }
            catch (Exception) { }
            finally { Volatile.Write(ref _readerDone, 1); }
        }

        void ErrorLoop(object state)
        {
            var proc = (System.Diagnostics.Process)state;
            try
            {
                for (;;)
                {
                    string line = proc.StandardError.ReadLine();
                    if (line == null) break;
                    if (line.Length > 0) Debug.Log("needle serve: " + line);
                }
            }
            catch (Exception) { }
        }

        void Dispatch(string line)
        {
            OutOfProcessReply reply;
            try
            {
                var parsed = JsonValue.Parse(line.AsSpan());
                var idNode = parsed.Get("id");
                var id = 0;
                if (idNode != null)
                {
                    if (idNode.Kind == JsonKind.Number) int.TryParse(idNode.Text, out id);
                    else if (idNode.Kind == JsonKind.String) int.TryParse(idNode.Text, out id);
                }
                var successNode = parsed.Get("success");
                var success = successNode != null && successNode.Kind == JsonKind.Bool && successNode.Boolean;
                reply = new OutOfProcessReply
                {
                    Id = id,
                    Success = success,
                    Type = parsed.GetString("type") == null ? string.Empty : parsed.GetString("type"),
                    Error = parsed.GetString("error") == null ? string.Empty : parsed.GetString("error"),
                    RawJson = line,
                };
            }
            catch (Exception e)
            {
                reply = new OutOfProcessReply { Id = 0, Success = false, Type = "error", Error = "unparseable reply: " + e.Message, RawJson = line };
            }
            lock (_gate) { _received++; }
            Action<OutOfProcessReply> handler = Replied;
            if (handler != null) handler(reply);
        }

        string ResolveTools()
        {
            if (Tools != null)
            {
                var temp = System.IO.Path.Combine(Application.temporaryCachePath, "needle-tools.json");
                System.IO.File.WriteAllText(temp, Tools.text);
                return temp;
            }
            if (!string.IsNullOrEmpty(ToolsFile))
            {
                var path = ResolvePath(ToolsFile);
                if (!System.IO.File.Exists(path)) throw new Exception(path);
                return path;
            }
            {
                var temp = System.IO.Path.Combine(Application.temporaryCachePath, "needle-tools.json");
                System.IO.File.WriteAllText(temp, "[]");
                return temp;
            }
        }

        static string ResolvePath(string name)
        {
            if (string.IsNullOrEmpty(name)) return name;
            if (System.IO.Path.IsPathRooted(name)) return name;
            return System.IO.Path.Combine(Application.streamingAssetsPath, name);
        }

        static string QuoteArg(string arg)
        {
            if (arg.IndexOfAny(new char[] { ' ', '\t', '"' }) < 0) return arg;
            return "\"" + arg.Replace("\"", "\\\"") + "\"";
        }

        static string EncodeRequest(int id, string type, string text)
        {
            var b = new StringBuilder(64 + text.Length);
            b.Append("{\"id\":").Append(id).Append(",\"type\":");
            JsonValue.WriteString(b, type);
            b.Append(",\"text\":");
            JsonValue.WriteString(b, text);
            b.Append('}');
            return b.ToString();
        }

        readonly object _gate = new object();
        readonly object _writeGate = new object();
        readonly Queue<string> _replies = new Queue<string>();
        System.Diagnostics.Process _proc;
        int _nextId = 1;
        int _sent;
        int _received;
        int _readerDone;
        int _shutdown;

        int MaxQueued = 256;
    }

    /// <summary>One server reply line, parsed. RawJson always carries the full line.</summary>
    [Serializable]
    public struct OutOfProcessReply
    {
        public int Id;
        public bool Success;
        public string Type;
        public string Error;
        public string RawJson;
    }
}
