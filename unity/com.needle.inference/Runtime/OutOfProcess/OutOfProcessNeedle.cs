using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Needle
{
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

        public event Action<OutOfProcessReply> Replied;

        public bool Connected { get; private set; }

        public int PendingRequests
        {
            get { lock (_gate) { return _sent - _received; } }
        }

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

        public int Ask(string query)
        {
            return Send("query", query);
        }

        public int SendToolResult(string resultJson)
        {
            return Send("tool_result", resultJson);
        }

        public int ResetServer()
        {
            return Send("reset", string.Empty);
        }

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
