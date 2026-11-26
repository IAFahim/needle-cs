using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

namespace Needle
{
    /// <summary>How an argument is handed to its event.</summary>
    public enum NeedleArgumentKind
    {
        Text,
        Number,
        WholeNumber,
        Flag,
    }

    /// <summary>A text value from a tool call.</summary>
    [Serializable] public sealed class NeedleTextEvent : UnityEvent<string> { }

    /// <summary>A number from a tool call.</summary>
    [Serializable] public sealed class NeedleNumberEvent : UnityEvent<float> { }

    /// <summary>A whole number from a tool call.</summary>
    [Serializable] public sealed class NeedleWholeNumberEvent : UnityEvent<int> { }

    /// <summary>A yes or no from a tool call.</summary>
    [Serializable] public sealed class NeedleFlagEvent : UnityEvent<bool> { }

    /// <summary>A tool call: the tool's name and its arguments as JSON.</summary>
    [Serializable] public sealed class NeedleCallEvent : UnityEvent<string, string> { }

    /// <summary>Hands one argument of a tool call to an event.</summary>
    [Serializable]
    public sealed class NeedleArgumentBinding
    {
        [Tooltip("The parameter's name in the tool set.")]
        public string Parameter = string.Empty;

        [Tooltip("What kind of value to hand on.")]
        public NeedleArgumentKind Kind = NeedleArgumentKind.Text;

        public NeedleTextEvent Text = new NeedleTextEvent();
        public NeedleNumberEvent Number = new NeedleNumberEvent();
        public NeedleWholeNumberEvent WholeNumber = new NeedleWholeNumberEvent();
        public NeedleFlagEvent Flag = new NeedleFlagEvent();
    }

    /// <summary>What happens in the scene when the agent calls one tool.</summary>
    [Serializable]
    public sealed class NeedleToolBinding
    {
        [Tooltip("The tool's name in the tool set.")]
        public string Tool = string.Empty;

        [Tooltip("Runs every time the agent calls this tool.")]
        public UnityEvent Called = new UnityEvent();

        [Tooltip("Runs with the call's arguments as JSON, for scripts that read them themselves.")]
        public NeedleTextEvent CalledWithArguments = new NeedleTextEvent();

        [Tooltip("Hands single arguments to their own events, after Called.")]
        public List<NeedleArgumentBinding> Arguments = new List<NeedleArgumentBinding>();
    }
}
