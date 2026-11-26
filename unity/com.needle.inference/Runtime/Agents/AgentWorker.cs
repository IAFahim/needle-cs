using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace Needle
{
    // Owns one NeedleAgent on a thread of its own, so loading, warming up and answering never run
    // on Unity's main thread. The main thread hands over one order at a time and collects finished
    // results; both sides only take a lock long enough to swap a reference.
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
        readonly BindingPlan[] plan;
        readonly Thread thread;
        readonly AutoResetEvent wake = new AutoResetEvent(false);
        readonly object gate = new object();
        readonly List<NeedleAgentResult> finished = new List<NeedleAgentResult>();

        NeedleAgent agent;
        string pending;
        volatile bool stopping;
        volatile bool running;
        volatile bool ready;
        volatile string loadError;
        bool modelAcquired;

        public AgentWorker(string modelPath, bool prefault, string toolsJson, AgentOptions options, string persona, bool warmUp, BindingPlan[] plan)
        {
            this.modelPath = modelPath;
            this.prefault = prefault;
            this.toolsJson = toolsJson;
            this.options = options;
            this.persona = persona;
            this.warmUp = warmUp;
            this.plan = plan ?? new BindingPlan[0];
            thread = new Thread(Run) { IsBackground = true, Name = "Needle agent" };
            thread.Start();
        }

        /// <summary>The agent is loaded (and warmed up, when asked) and takes orders.</summary>
        public bool IsReady => ready;

        /// <summary>Why the agent could not load, or null.</summary>
        public string LoadError => loadError;

        /// <summary>An order is waiting or being answered.</summary>
        public bool IsBusy
        {
            get
            {
                if (running) return true;
                lock (gate) return pending != null;
            }
        }

        /// <summary>Queues an order; one that has not started yet is replaced.</summary>
        public void Submit(string order)
        {
            lock (gate) pending = order ?? string.Empty;
            wake.Set();
        }

        /// <summary>Stops the order being answered. Safe from any thread.</summary>
        public void CancelRunning()
        {
            if (running && ready) agent.Cancel();
        }

        /// <summary>Drops an order that has not started yet.</summary>
        public void ClearPending()
        {
            lock (gate) pending = null;
        }

        /// <summary>Moves the finished results into <paramref name="into"/>.</summary>
        public void Drain(List<NeedleAgentResult> into)
        {
            lock (gate)
            {
                if (finished.Count == 0) return;
                into.AddRange(finished);
                finished.Clear();
            }
        }

        void Run()
        {
            try
            {
                NeedleModel model = NeedleModelAsset.Acquire(modelPath, prefault);
                modelAcquired = true;
                agent = model.Agent(toolsJson, options);
                if (!string.IsNullOrWhiteSpace(persona)) agent.SetSystem(persona);
                if (warmUp) agent.WarmUp();
                ready = true;
            }
            catch (Exception e)
            {
                loadError = e.Message;
                Close();
                return;
            }

            while (!stopping)
            {
                wake.WaitOne();
                while (!stopping)
                {
                    string order;
                    lock (gate)
                    {
                        order = pending;
                        pending = null;
                        if (order != null) running = true;
                    }
                    if (order == null) break;
                    NeedleAgentResult result = AnswerOrder(order);
                    lock (gate)
                    {
                        finished.Add(result);
                        running = false;
                    }
                }
            }
            Close();
        }

        NeedleAgentResult AnswerOrder(string order)
        {
            var result = new NeedleAgentResult { Order = order, Text = string.Empty, Calls = new NeedleAgentCall[0] };
            try
            {
                Answer answer = agent.Ask(order);
                result.Kind = answer.Kind;
                result.Why = answer.Why;
                result.Confidence = answer.Confidence;
                result.Milliseconds = answer.Details.ElapsedTicks * 1000.0 / Stopwatch.Frequency;
                if (answer.Kind == AnswerKind.Text) result.Text = answer.Text.ToString();
                if (answer.Kind == AnswerKind.Calls)
                {
                    CallList calls = answer.Calls;
                    var copied = new NeedleAgentCall[calls.Count];
                    for (int i = 0; i < calls.Count; i++) copied[i] = Copy(calls[i]);
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

        NeedleAgentCall Copy(Call call)
        {
            var copy = new NeedleAgentCall { Tool = call.Name.ToString(), Arguments = call.Arguments.ToString() };
            for (int b = 0; b < plan.Length; b++)
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

        void Close()
        {
            ready = false;
            try { agent.Dispose(); } catch (Exception) { }
            if (modelAcquired) NeedleModelAsset.Release(modelPath);
            modelAcquired = false;
        }

        public void Dispose()
        {
            if (stopping) return;
            stopping = true;
            CancelRunning();
            wake.Set();
            if (!thread.Join(TimeSpan.FromSeconds(10)))
                UnityEngine.Debug.LogWarning("[Needle] An agent's thread did not stop within ten seconds; it closes when its answer finishes.");
            else
                wake.Dispose();
        }
    }
}
