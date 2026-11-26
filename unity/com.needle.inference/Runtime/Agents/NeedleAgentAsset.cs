using UnityEngine;

namespace Needle
{
    /// <summary>
    /// Everything an agent is: which model it runs, what it can do, what it always knows, and how
    /// careful it is. Several characters can share one asset.
    /// </summary>
    [CreateAssetMenu(menuName = "Needle/Agent", fileName = "Agent", order = 102)]
    public sealed class NeedleAgentAsset : ScriptableObject
    {
        [Tooltip("The model this agent runs.")]
        public NeedleModelAsset Model;

        [Tooltip("The actions this agent can take.")]
        public NeedleToolSet Tools;

        [Tooltip("Facts the agent always knows, one per line, for example who it is, where it is, or who the player is.")]
        [TextArea(3, 10)]
        public string Persona = string.Empty;

        [Header("Behaviour")]
        [Tooltip("Only act when the model is at least this sure. Below it the agent says it is unsure instead. 0.3 suits most games.")]
        [Range(0f, 1f)]
        public float RefuseBelow = 0.3f;

        [Tooltip("What the agent says when it is unsure or cannot help.")]
        public string UnsureLine = "Sorry, I didn't catch that.";

        [Tooltip("Remember earlier orders, so \"do that again\" or \"him too\" work. Off: every order stands alone, which is faster and more predictable.")]
        public bool Remember;

        [Tooltip("Let the model think briefly before it acts. Slower; helps with long or tricky orders.")]
        public bool Think = true;

        [Tooltip("Add today's date and time to what the agent knows.")]
        public bool KnowsTheDate = true;

        [Header("Performance")]
        [Tooltip("CPU threads the agent uses while it answers. 2 leaves room for the game; more answers faster.")]
        [Range(1, 8)]
        public int Threads = 2;

        [Tooltip("With many tools, show the model only the best matching ones. 0 shows every tool.")]
        [Min(0)]
        public int OnlyBestTools;

        [Tooltip("Load and warm the agent up as soon as it is enabled, so the first order is answered at full speed.")]
        public bool WarmUp = true;

        /// <summary>The engine options this asset stands for.</summary>
        public AgentOptions ToOptions()
        {
            return new AgentOptions
            {
                Threading = ThreadPolicy.Game.WithThreads(Mathf.Max(1, Threads)),
                RefuseBelow = RefuseBelow <= 0f ? -1f : RefuseBelow,
                Stateless = !Remember,
                ReasoningBudget = Think ? 0 : AgentOptions.NoReasoning,
                AutoDate = KnowsTheDate ? Toggle.On : Toggle.Off,
                RetrievalTopK = OnlyBestTools,
            };
        }
    }
}
