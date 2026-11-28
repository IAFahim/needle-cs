namespace Needle;

/// <summary>A three-way switch whose <see cref="Default"/> means "do what upstream Needle 4.x does".</summary>
public enum Toggle : byte
{
    Default,
    On,
    Off,
}

/// <summary>
/// How an agent runs. Every member's <c>default</c> means what upstream Needle 4.x does, so
/// <c>default(AgentOptions)</c> is the right choice for most apps. The struct holds no references.
/// </summary>
public struct AgentOptions
{
    /// <summary>Use with <see cref="ReasoningBudget"/> to skip thinking entirely.</summary>
    public const int NoReasoning = -1;

    /// <summary>Upstream 4.x confidence floor: calls below it are suppressed, not returned.</summary>
    public const float UpstreamFloor = 0.1f;

    /// <summary>Worker threads. 0 picks the physical performance cores, up to 4.</summary>
    public int Threads { get; set; }

    /// <summary>Full thread policy (priority, placement, spin). Overrides <see cref="Threads"/> when set.</summary>
    public ThreadPolicy? Threading { get; set; }

    /// <summary>Caps the work a single turn may do before it gives the thread back. Unlimited by default.</summary>
    public FrameBudget FrameBudget { get; set; }

    /// <summary>Context window in tokens. 0 means 2048.</summary>
    public int Context { get; set; }

    /// <summary>Tokens the model may think before it answers. 0 means 96; <see cref="NoReasoning"/> means none.</summary>
    public int ReasoningBudget { get; set; }

    /// <summary>Token budget for the calls of one turn. 0 means 384.</summary>
    public int MaxCallTokens { get; set; }

    /// <summary>Confidence floor. 0 means the upstream floor of 0.1; a negative value never suppresses on confidence.</summary>
    public float RefuseBelow { get; set; }

    /// <summary>Adds "date: YYYY-MM-DD Ddd HH:MM" to the system facts. On by default, as in 4.x.</summary>
    public Toggle AutoDate { get; set; }

    /// <summary>4.x argument repair and the grounding gate. On by default.</summary>
    public Toggle Repair { get; set; }

    /// <summary>Forget the conversation after every turn.</summary>
    public bool Stateless { get; set; }

    /// <summary>Show the model only the best matching tools. 0 means every tool.</summary>
    public int RetrievalTopK { get; set; }

    /// <summary>Feed grammar-forced tokens in one pass. Lossless. Off by default for now.</summary>
    public Toggle JumpForward { get; set; }

    /// <summary>Draft call tokens from the prompt and verify them in one pass. Lossless. Off by default for now.</summary>
    public Toggle PromptLookup { get; set; }

    internal readonly NeedleOptions ToNeedleOptions(string? system) => new()
    {
        Threads = Threads <= 0 ? 0 : Threads,
        Threading = Threading,
        FrameBudget = FrameBudget,
        Context = Context > 0 ? Context : 2048,
        ReasoningBudget = ReasoningBudget == 0 ? 96 : ReasoningBudget < 0 ? 0 : ReasoningBudget,
        MaxCallTokens = MaxCallTokens > 0 ? MaxCallTokens : 384,
        System = system,
        AutoDate = AutoDate != Toggle.Off,
        RefuseBelow = RefuseBelow == 0 ? UpstreamFloor : RefuseBelow < 0 ? 0f : RefuseBelow,
        RetrievalTopK = RetrievalTopK > 0 ? RetrievalTopK : int.MaxValue,
        JumpForward = JumpForward == Toggle.On,
        PromptLookup = PromptLookup == Toggle.On,
        Stateless = Stateless,
        Repair = Repair != Toggle.Off,
    };
}
