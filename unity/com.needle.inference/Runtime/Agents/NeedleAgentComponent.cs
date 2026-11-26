using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

namespace Needle
{
    /// <summary>What an agent does with an order that arrives while it is still answering the last one.</summary>
    public enum NeedleBusyPolicy
    {
        /// <summary>Answer it next; a newer order replaces one still waiting.</summary>
        AnswerLatest,
        /// <summary>Drop it and run On Busy.</summary>
        Ignore,
        /// <summary>Stop the current answer and take the new order instead.</summary>
        Interrupt,
    }

    /// <summary>
    /// Turns plain-language orders into actions in the scene. Give it an Agent asset, call
    /// <see cref="Ask"/> (from a script, a button or an input field's On End Edit), and wire each
    /// tool to what it should do. The model loads and answers on its own thread; the events run on
    /// the main thread, in Update, after the answer is ready.
    /// </summary>
    [AddComponentMenu("Needle/Needle Agent")]
    [DisallowMultipleComponent]
    public sealed class NeedleAgentComponent : MonoBehaviour
    {
        [Tooltip("The agent's model, tools and personality.")]
        public NeedleAgentAsset Agent;

        [Tooltip("What happens in the scene when the agent calls each tool.")]
        public List<NeedleToolBinding> Tools = new List<NeedleToolBinding>();

        [Tooltip("What to do with an order that arrives while the agent is still answering.")]
        public NeedleBusyPolicy WhenBusy = NeedleBusyPolicy.AnswerLatest;

        [Tooltip("Print every order and answer to the Console.")]
        public bool LogAnswers;

        [Header("Events")]
        [Tooltip("The agent has loaded and takes orders.")]
        public UnityEvent OnReady = new UnityEvent();

        [Tooltip("Every tool call, with the tool's name and its arguments as JSON, after the tool's own events.")]
        public NeedleCallEvent OnCall = new NeedleCallEvent();

        [Tooltip("The agent answered in words instead of acting.")]
        public NeedleTextEvent OnReply = new NeedleTextEvent();

        [Tooltip("The agent was not sure enough to act, or chose not to. Gets the Agent asset's unsure line.")]
        public NeedleTextEvent OnUnsure = new NeedleTextEvent();

        [Tooltip("The order could not be answered, or the agent could not load. Gets the reason.")]
        public NeedleTextEvent OnFailed = new NeedleTextEvent();

        [Tooltip("An order was dropped because the agent was busy (When Busy: Ignore).")]
        public NeedleTextEvent OnBusy = new NeedleTextEvent();

        /// <summary>Every answer, for scripts; runs before the events above.</summary>
        public event Action<NeedleAgentResult> Answered;

        /// <summary>The agent has loaded and takes orders.</summary>
        public bool IsReady => worker != null && worker.IsReady;

        /// <summary>An order is waiting or being answered.</summary>
        public bool IsBusy => worker != null && worker.IsBusy;

        /// <summary>The last answer, or null.</summary>
        public NeedleAgentResult LastResult { get; private set; }

        AgentWorker worker;
        string queuedBeforeReady;
        bool announcedReady;
        bool failedToLoad;
        readonly List<NeedleAgentResult> drained = new List<NeedleAgentResult>();

        void OnEnable()
        {
            announcedReady = false;
            failedToLoad = false;
            if (Agent == null) { Fail("This Needle Agent has no Agent asset."); return; }
            if (Agent.Model == null) { Fail("The Agent asset '" + Agent.name + "' has no model."); return; }
            if (Agent.Tools == null) { Fail("The Agent asset '" + Agent.name + "' has no tool set."); return; }
            StartCoroutine(Load());
        }

        void OnDisable()
        {
            StopAllCoroutines();
            if (worker != null) worker.Dispose();
            worker = null;
        }

        IEnumerator Load()
        {
            string path = null, error = null;
            yield return Agent.Model.Prepare(p => path = p, e => error = e);
            if (error != null) { Fail(error); yield break; }
            worker = new AgentWorker(path, Agent.Model.Prefault, Agent.Tools.ToJson(), Agent.ToOptions(), Agent.Persona, Agent.WarmUp, Plan());
        }

        AgentWorker.BindingPlan[] Plan()
        {
            var plan = new AgentWorker.BindingPlan[Tools.Count];
            for (int b = 0; b < Tools.Count; b++)
            {
                NeedleToolBinding binding = Tools[b];
                int n = binding.Arguments.Count;
                plan[b] = new AgentWorker.BindingPlan
                {
                    Tool = string.IsNullOrEmpty(binding.Tool) ? null : binding.Tool,
                    Parameters = new string[n],
                    Kinds = new NeedleArgumentKind[n],
                };
                for (int a = 0; a < n; a++)
                {
                    plan[b].Parameters[a] = binding.Arguments[a].Parameter;
                    plan[b].Kinds[a] = binding.Arguments[a].Kind;
                }
            }
            return plan;
        }

        /// <summary>
        /// Gives the agent an order in plain language, for example "move to the tower". The answer
        /// arrives through the events a few frames later; an order given while the agent loads is
        /// answered once it is ready.
        /// </summary>
        public void Ask(string order)
        {
            if (string.IsNullOrWhiteSpace(order)) return;
            if (failedToLoad) { OnFailed.Invoke("The agent did not load, so it cannot take orders."); return; }
            if (worker == null || !worker.IsReady)
            {
                queuedBeforeReady = order;
                return;
            }
            if (worker.IsBusy)
            {
                switch (WhenBusy)
                {
                    case NeedleBusyPolicy.Ignore:
                        OnBusy.Invoke(order);
                        return;
                    case NeedleBusyPolicy.Interrupt:
                        worker.ClearPending();
                        worker.CancelRunning();
                        break;
                }
            }
            worker.Submit(order);
        }

        /// <summary>Stops the answer in progress and drops a waiting order.</summary>
        public void Cancel()
        {
            queuedBeforeReady = null;
            if (worker == null) return;
            worker.ClearPending();
            worker.CancelRunning();
        }

        void Update()
        {
            if (worker == null) return;
            if (!announcedReady)
            {
                if (worker.LoadError != null) { Fail("The agent could not load: " + worker.LoadError); worker.Dispose(); worker = null; return; }
                if (!worker.IsReady) return;
                announcedReady = true;
                OnReady.Invoke();
                if (queuedBeforeReady != null)
                {
                    string order = queuedBeforeReady;
                    queuedBeforeReady = null;
                    worker.Submit(order);
                }
            }
            worker.Drain(drained);
            for (int i = 0; i < drained.Count; i++) Dispatch(drained[i]);
            drained.Clear();
        }

        void Dispatch(NeedleAgentResult result)
        {
            LastResult = result;
            if (LogAnswers) Debug.Log("[Needle] \"" + result.Order + "\" -> " + result + " (" + result.Confidence.ToString("0.00") + ", " + result.Milliseconds.ToString("0") + " ms)", this);
            Action<NeedleAgentResult> answered = Answered;
            if (answered != null) answered(result);
            switch (result.Kind)
            {
                case AnswerKind.Calls:
                    for (int i = 0; i < result.Calls.Length; i++) Invoke(result.Calls[i]);
                    break;
                case AnswerKind.Text:
                    OnReply.Invoke(result.Text);
                    break;
                case AnswerKind.Refused:
                case AnswerKind.Declined:
                    OnUnsure.Invoke(Agent != null ? Agent.UnsureLine : string.Empty);
                    break;
                case AnswerKind.Cancelled:
                    break;
                default:
                    OnFailed.Invoke(string.IsNullOrEmpty(result.Text) ? result.Why.ToString() : result.Text);
                    break;
            }
        }

        void Invoke(NeedleAgentCall call)
        {
            if (call.Binding >= 0 && call.Binding < Tools.Count)
            {
                NeedleToolBinding binding = Tools[call.Binding];
                binding.Called.Invoke();
                binding.CalledWithArguments.Invoke(call.Arguments);
                int n = Math.Min(binding.Arguments.Count, call.Found == null ? 0 : call.Found.Length);
                for (int a = 0; a < n; a++)
                {
                    if (!call.Found[a]) continue;
                    NeedleArgumentBinding argument = binding.Arguments[a];
                    switch (argument.Kind)
                    {
                        case NeedleArgumentKind.Number: argument.Number.Invoke((float)call.Numbers[a]); break;
                        case NeedleArgumentKind.WholeNumber: argument.WholeNumber.Invoke((int)Math.Round(call.Numbers[a])); break;
                        case NeedleArgumentKind.Flag: argument.Flag.Invoke(call.Flags[a]); break;
                        default: argument.Text.Invoke(call.Texts[a]); break;
                    }
                }
            }
            OnCall.Invoke(call.Tool, call.Arguments);
        }

        void Fail(string message)
        {
            failedToLoad = true;
            Debug.LogError("[Needle] " + message, this);
            OnFailed.Invoke(message);
        }
    }
}
