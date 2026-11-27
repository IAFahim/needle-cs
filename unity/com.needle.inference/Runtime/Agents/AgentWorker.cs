using System;
using System.Diagnostics;
using System.Threading;

namespace Needle
{
    // Loads one NeedleAgent off the main thread (opening the model, building the agent and warming
    // it up can take a while), then drives it through the engine's own game path: TryPost starts an
    // answer on the agent's thread and returns at once, TryTake picks it up on a later frame. Results
    // are copied out of the engine's buffers on the main thread, into objects that stay valid.
    internal sealed class AgentWorker : IDisposable
    {
        internal struct BindingPlan
        {
            public string Tool;
            public string[] Parameters;
            public NeedleArgumentKind[] Kinds;
        }

        readonly string modelPath;
        readonly bool prefault;
        readonly string toolsJson;
        readonly AgentOptions options;
        readonly string persona;
        readonly bool warmUp;
        readonly object gate = new object();
        readonly Thread loader;

        NeedleAgent agent;
        bool modelAcquired;
        bool disposed;
        volatile bool ready;
        volatile string loadError;
        string posted;

        public AgentWorker(string modelPath, bool prefault, string toolsJson, AgentOptions options, string persona, bool warmUp)
        {
            this.modelPath = modelPath;
            this.prefault = prefault;
            this.toolsJson = toolsJson;
            this.options = options;
            this.persona = persona;
            this.warmUp = warmUp;
            loader = new Thread(Load) { IsBackground = true, Name = "Needle agent loader" };
            loader.Start();
        }

        /// <summary>The agent is loaded (and warmed up, when asked) and takes orders.</summary>
        public bool IsReady => ready;

        /// <summary>Why the agent could not load, or null.</summary>
        public string LoadError => loadError;

        /// <summary>An order is being answered.</summary>
        public bool IsBusy => ready && agent.IsBusy;

        /// <summary>Starts answering an order on the agent's thread; false while another is in flight.</summary>
        public bool TryPost(string order)
        {
            if (!ready || !agent.TryPost(order)) return false;
            posted = order;
            return true;
        }

        /// <summary>Stops the answer in flight; it comes back as Cancelled.</summary>
        public void CancelRunning()
        {
            if (ready && agent.IsBusy) agent.Cancel();
        }

        /// <summary>Picks up a finished answer, copied so it stays valid.</summary>
        public bool TryTake(BindingPlan[] plan, out NeedleAgentResult result)
        {
            result = null;
            Answer answer;
            if (!ready || !agent.TryTake(out answer)) return false;
            result = Copy(answer, posted, plan);
            posted = null;
            return true;
        }

        void Load()
        {
            NeedleAgent loaded = default;
            bool acquired = false;
            try
            {
                NeedleModel model = NeedleModelAsset.Acquire(modelPath, prefault);
                acquired = true;
                loaded = model.Agent(toolsJson, options);
                if (!string.IsNullOrWhiteSpace(persona)) loaded.SetSystem(persona);
                if (warmUp) loaded.WarmUp();
            }
            catch (Exception e)
            {
                if (loaded.IsOpen) loaded.Dispose();
                if (acquired) NeedleModelAsset.Release(modelPath);
                loadError = e.Message;
                return;
            }
            lock (gate)
            {
                if (!disposed)
                {
                    agent = loaded;
                    modelAcquired = true;
                    ready = true;
                    return;
                }
            }
            // Disposed while loading: nobody will use this agent.
            loaded.Dispose();
            NeedleModelAsset.Release(modelPath);
        }

        static NeedleAgentResult Copy(Answer answer, string order, BindingPlan[] plan)
        {
            var result = new NeedleAgentResult { Order = order ?? string.Empty, Text = string.Empty, Calls = Empty };
            try
            {
                result.Kind = answer.Kind;
                result.Why = answer.Why;
                result.Confidence = answer.Confidence;
                result.Milliseconds = answer.Details.ElapsedTicks * 1000.0 / Stopwatch.Frequency;
                if (answer.Kind == AnswerKind.Text) result.Text = answer.Text.ToString();
                if (answer.Kind == AnswerKind.Calls)
                {
                    CallList calls = answer.Calls;
                    var copied = new NeedleAgentCall[calls.Count];
                    for (int i = 0; i < calls.Count; i++) copied[i] = Copy(calls[i], plan);
                    result.Calls = copied;
                }
            }
            catch (Exception e)
            {
                result.Kind = AnswerKind.Failed;
                result.Why = AnswerWhy.Failed;
                result.Text = e.Message;
            }
            return result;
        }

        static readonly NeedleAgentCall[] Empty = new NeedleAgentCall[0];

        static NeedleAgentCall Copy(Call call, BindingPlan[] plan)
        {
            var copy = new NeedleAgentCall { Tool = call.Name.ToString(), Arguments = call.Arguments.ToString() };
            for (int b = 0; plan != null && b < plan.Length; b++)
            {
                if (plan[b].Tool == null || !call.Is(plan[b].Tool)) continue;
                copy.Binding = b;
                int n = plan[b].Parameters.Length;
                copy.Texts = new string[n];
                copy.Numbers = new double[n];
                copy.Flags = new bool[n];
                copy.Found = new bool[n];
                for (int p = 0; p < n; p++)
                {
                    string name = plan[b].Parameters[p];
                    if (string.IsNullOrEmpty(name)) continue;
                    switch (plan[b].Kinds[p])
                    {
                        case NeedleArgumentKind.Number:
                        case NeedleArgumentKind.WholeNumber:
                            double number;
                            copy.Found[p] = call.TryGetDouble(name, out number);
                            copy.Numbers[p] = number;
                            break;
                        case NeedleArgumentKind.Flag:
                            bool flag;
                            copy.Found[p] = call.TryGetBoolean(name, out flag);
                            copy.Flags[p] = flag;
                            break;
                        default:
                            string text = call.GetString(name);
                            Utf8Span raw;
                            if (text == null && call.TryGet(name, out raw)) text = raw.ToString();
                            copy.Found[p] = text != null;
                            copy.Texts[p] = text;
                            break;
                    }
                }
                break;
            }
            return copy;
        }

        public void Dispose()
        {
            lock (gate)
            {
                if (disposed) return;
                disposed = true;
                if (!ready) return;
                ready = false;
            }
            // Dispose cancels an answer in flight and waits for the agent's thread to stop.
            try { agent.Dispose(); } catch (Exception) { }
            if (modelAcquired) NeedleModelAsset.Release(modelPath);
            modelAcquired = false;
        }
    }
}
