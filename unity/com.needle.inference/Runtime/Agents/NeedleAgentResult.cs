namespace Needle
{
    /// <summary>One tool call of a <see cref="NeedleAgentResult"/>, copied out of the engine's buffers.</summary>
    public sealed class NeedleAgentCall
    {
        /// <summary>The tool's name.</summary>
        public string Tool { get; internal set; }

        /// <summary>The arguments as JSON.</summary>
        public string Arguments { get; internal set; }

        internal int Binding = -1;
        internal string[] Texts;
        internal double[] Numbers;
        internal bool[] Flags;
        internal bool[] Found;

        public override string ToString() => Tool + "(" + Arguments + ")";
    }

    /// <summary>What an agent made of one order, safe to keep after the next one.</summary>
    public sealed class NeedleAgentResult
    {
        /// <summary>The order as it was asked.</summary>
        public string Order { get; internal set; }

        /// <summary>What the model did.</summary>
        public AnswerKind Kind { get; internal set; }

        /// <summary>Why it is not a call or a reply, when it is not.</summary>
        public AnswerWhy Why { get; internal set; }

        /// <summary>How sure the model was, from 0 to 1.</summary>
        public float Confidence { get; internal set; }

        /// <summary>What the model said instead of calling a tool, or empty.</summary>
        public string Text { get; internal set; }

        /// <summary>The tool calls, in order; empty unless <see cref="Kind"/> is <see cref="AnswerKind.Calls"/>.</summary>
        public NeedleAgentCall[] Calls { get; internal set; }

        /// <summary>How long the engine took, in milliseconds.</summary>
        public double Milliseconds { get; internal set; }

        /// <summary>The model acted or replied.</summary>
        public bool Ok => Kind == AnswerKind.Calls || Kind == AnswerKind.Text;

        public override string ToString()
        {
            if (Kind == AnswerKind.Calls) return string.Join<NeedleAgentCall>(", ", Calls);
            if (Kind == AnswerKind.Text) return Text;
            return Kind + (Why == AnswerWhy.None ? string.Empty : " (" + Why + ")");
        }
    }
}
