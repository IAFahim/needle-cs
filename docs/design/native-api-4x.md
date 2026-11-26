# needle-cs 4.x: the native API plan

Status: proposal for review. Nothing below past phase 0 is built yet.
Branch: `claude/engine-4x`.

## 1. The goal in one sentence

Needle should read like a part of C# itself, in the way `Span<T>`, `Utf8JsonReader` and `System.Numerics` do: the C# compiler sees your tools, the runtime never sees a garbage-collected object, and a designer can go from "I have an idea" to "the NPC obeys me" in five lines.

## 2. What "no managed types" means here

This is our reading. Please correct it if it is wrong, because everything below depends on it.

1. **The core API surface has no GC types.** No `class`, `string`, `delegate`, `object`, array, `List<T>`, `Task` or exception on any call you make per turn. Handles are `struct`s, views are `ref struct`s, text goes in and out as UTF-8 spans, tool arguments are unmanaged structs.
2. **The engine's state lives off the GC heap.** Today `NeedleSession` is a class that owns some managed arrays. In the new design an agent is one block of native memory (an arena) and the handle you hold is a pointer-sized struct. The GC never scans it, never moves it, never pauses for it.
3. **No reflection, boxing, JSON strings or delegates on the hot path.** Tool binding happens at compile time through a source generator. The model's output is decoded straight into your structs.
4. **The same code runs under NativeAOT, trimming, IL2CPP and Burst.** No feature that any of those four rejects.
5. **Managed conveniences still exist, but outside the core.** `async`/`await`, `IAsyncEnumerable`, `IChatClient` (Microsoft.Extensions.AI) and dependency injection come as thin optional packages built on top. You never pay for them unless you reference them.
6. **What we do not mean:** we are not writing C, and we are not hiding behind `IntPtr`. Everything is idiomatic, type-safe C#: generics, pattern matching, `using`, `foreach`, `u8` literals, interpolated strings. "Unsafe" stays inside the library.

## 3. Before and after

### Before (today)

```csharp
using var model = Model.Open("needle3.cact");
using var session = new NeedleSession(model, toolsJson, new NeedleOptions { Threads = 4 });

CallResult result = session.Complete("set a timer for 12 minutes");   // allocates strings, lists, records
foreach (FunctionCall call in result.FunctionCalls)
    if (call.Name == "set_timer")
        Start(JsonDocument.Parse(call.ArgumentsJson).RootElement.GetProperty("minutes").GetInt32());
```

### After

```csharp
[Tools]
public partial struct Squad
{
    [Tool("Move the squad to a map position.")]       public partial record struct Move(float X, float Y);
    [Tool("Attack a named target.")]                   public partial record struct Attack(Text32 Target);
    [Tool("Give items to a squad member.")]            public partial record struct Give(Text32 Member, Item Item, [Range(1, 99)] int Count);
    [Tool("Change formation.")]                        public partial record struct Formation(Shape Shape);

    public enum Item : byte { Potion, Arrow, Gold }
    public enum Shape : byte { Line, Wedge, Circle }
}

using var model = NeedleModel.Open("needle3.cact");
using var agent = model.Agent<Squad>();

foreach (var call in agent.Ask("give 3 potions to Lyra then form a wedge"))
{
    switch (call.Tool)
    {
        case Squad.Tool.Give:      Inventory.Give(call.Give.Member, call.Give.Item, call.Give.Count); break;
        case Squad.Tool.Formation: Formations.Set(call.Formation.Shape); break;
    }
}
```

That loop allocates zero bytes. A `Squad` value *is* one call: the generator turns the struct into a tagged union of its tools, so `call.Tool` says which one and `call.Give` is a `ref readonly` to the arguments, written directly by the decoder while the model was generating. `agent.Ask` returns a `ref struct` view over the agent's arena, and `Lyra` sits in an inline 32-byte UTF-8 buffer (`Text32`), not a string.

Interpolation works without allocating too, because `Ask` takes an interpolated string handler that writes UTF-8 straight into the agent's input buffer:

```csharp
agent.Ask($"{player.Name} says: {line}");
```

## 4. The layers

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Optional, managed: Needle.Extensions.AI (IChatClient), Needle.Hosting     │
 │ (AddNeedle, IOptions), async/await and IAsyncEnumerable adapters          │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ Core, unmanaged: NeedleModel, Agent<T>, Agent (JSON tools), Turn, Answer, │
 │ CallView, AgentHost + Mailbox (threads, games, Burst), SpeechModel,       │
 │ Transcriber, VoiceAgent<T>                                                │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ Compile time: Needle.Generators. [Tools] → schema, byte-level grammar,    │
 │ call union, decoder, trigger DFAs, validation tables, all as static data  │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ Engine: kernels (AVX-512/VNNI, AVX2, NEON dot, scalar, Burst intrinsics), │
 │ grammar automaton, 4.x repair and validation, speculative decoding        │
 └──────────────────────────────────────────────────────────────────────────┘
```

## 5. Key types

Signatures are illustrative; names are open to taste.

### Model

```csharp
public readonly struct NeedleModel : IDisposable
{
    public static NeedleModel Open(ReadOnlySpan<char> path, in ModelOptions options = default);     // memory-mapped, shared
    public static bool TryOpen(ReadOnlySpan<char> path, out NeedleModel model, out NeedleError error);
    public static unsafe NeedleModel FromMemory(byte* data, nuint length);                         // Unity NativeArray, embedded resources
    public Agent<TTools> Agent<TTools>(in AgentOptions options = default) where TTools : unmanaged, ITools<TTools>;
    public Agent Agent(ReadOnlySpan<byte> toolsJson, in AgentOptions options = default);           // tools known only at run time
    public ModelInfo Info { get; }
    public void Dispose();
}
```

Every throwing call has a `Try` twin that returns a `NeedleError` value instead, so code that must never allocate (a Burst job, a frame loop) can avoid exceptions entirely.

### Options

```csharp
public struct AgentOptions
{
    public Threads Threads;          // Threads.Auto (4.x semantics), Threads.Exactly(4), Threads.Game (low spin, never the caller)
    public int Context;              // 0 = model default
    public Reasoning Reasoning;      // Reasoning.Off, Reasoning.Budget(96), Reasoning.Auto
    public float RefuseBelow;        // 4.x confidence floor
    public Repair Repair;            // Repair.On (4.x default), Repair.Off
    public Utf8Span System;          // system facts, e.g. "date: 2026-11-26"u8
    public DateStamp Today;          // fills the date fact, 4.x auto_date
    public Memory Memory;            // Memory.Conversation or Memory.Stateless
}
```

All of it is unmanaged and `default` means "do what 4.x does".

### Agent and answers

```csharp
public struct Agent<TTools> : IDisposable where TTools : unmanaged, ITools<TTools>
{
    public Answer<TTools> Ask(ReadOnlySpan<byte> utf8);
    public Answer<TTools> Ask(ReadOnlySpan<char> text);
    public Answer<TTools> Ask([InterpolatedStringHandlerArgument("")] ref AskHandler text);
    public Answer<TTools> Reply<TResult>(in TResult toolResult) where TResult : unmanaged;      // tool result, multi-turn
    public Turn<TTools> Begin(ReadOnlySpan<byte> utf8);                                          // streaming and frame slicing
    public void Forget();                                                                        // reset the conversation
    public void Cancel();
}

public readonly ref struct Answer<TTools> where TTools : unmanaged, ITools<TTools>
{
    public AnswerKind Kind { get; }                 // Calls, Text, Declined, Refused, Failed
    public ReadOnlySpan<TTools> Calls { get; }      // each TTools value is one decoded call, in order
    public ReadOnlySpan<TTools> Suppressed { get; }
    public Utf8Span Text { get; }                   // what the model said instead of calling
    public Utf8Span Thinking { get; }
    public Confidence Confidence { get; }           // head, decode, combined
    public Validation Validation { get; }           // 4.x: Repaired, Negated, Invalid, Ungrounded (tool, field) pairs
    public Timing Timing { get; }                   // prefill, first token, decode, total, tokens
    public Failure Failure { get; }
    public Enumerator GetEnumerator();              // foreach over Calls
}
```

`Utf8Span` is a `ref struct` wrapper over `ReadOnlySpan<byte>` that formats into `Span<char>`, compares with `"…"u8`, and only makes a `string` when you call `ToString()`.

### Tools, generated at compile time

```csharp
public interface ITools<TSelf> where TSelf : unmanaged, ITools<TSelf>
{
    ReadOnlySpan<byte> Schema { get; }          // JSON schema the model sees, as a u8 constant
    ToolGrammar Grammar { get; }                // precompiled byte automaton
    // plus the generated union layout, Tool enum, decoder, validation and trigger tables
}
```

The interface uses instance members on a zero-size struct, not static abstract members, on purpose: Unity's compiler is C# 9 and its runtimes lack static abstracts, while `default(TTools).Grammar` compiles to a constant load on every runtime and is specialized by Burst.

What the generator gives you, all as static read-only data:

- the JSON schema and the byte-level grammar (already done today for `[NeedleTools]`),
- the union layout of `Squad` itself, its tool structs overlapping behind a `Tool` tag, so `call.Give` is a `ref readonly` with no copy and no boxing,
- a streaming decoder that fills those structs token by token, so the call is ready the instant the closing brace is sampled,
- range, enum and required-field tables for the 4.x validation stage,
- 4.x triggers compiled to DFAs at build time (`[Tool(..., Trigger = @"\b(heal|medic)\b")]`), matched with no regex engine at run time,
- analyzers that explain mistakes in the IDE: a `string` field ("use Text32 or Text64"), a missing description, an enum with spaces, a tool the model will confuse with another one.

Text fields use fixed-capacity inline UTF-8 types: `Text16`, `Text32`, `Text64`, `Text128`. The grammar caps generation at that length, so text can never overflow. Long free text belongs in `Answer.Text`, which points into the arena.

### Untyped tools, for tools that only exist at run time

```csharp
using var agent = model.Agent(toolsJsonUtf8);
foreach (var call in agent.Ask("turn off the kitchen light"u8))
{
    call.Name;                                    // Utf8Span: "turn_off_light"
    call.Arguments;                               // Utf8Span: {"room":"kitchen"}, 4.x-repaired
    call.TryGet("room"u8, out Utf8Span room);     // no parse tree, reads the spans the decoder recorded
}
```

### Streaming and frame budgets

A turn can run all at once or in slices you control, which is what a game loop or a UI thread wants:

```csharp
var turn = agent.Begin(query);
// every frame:
while (turn.Step(budget: TimeSpan.FromMilliseconds(2), out Token token))
{
    if (token.Kind == TokenKind.Thinking) ui.ShowThinking(token.Utf8);
    if (token.Kind == TokenKind.CallReady) Apply(token.Call);       // act on call 1 while call 2 is still generating
}
if (turn.IsDone) Finish(turn.Answer);
```

`Step` is pull-based, allocation-free and never blocks.

### Threads, hosts and the game thread

`AgentHost` owns an agent on its own worker thread and talks over `Mailbox`, the lock-free unmanaged ring that exists today as `NeedleChannel`. Submit and receive are pointer arithmetic plus `Interlocked`, so a Burst job can call them.

```csharp
using var host = AgentHost.Start<Squad>(model, options);
host.Mailbox.Send("hold the bridge"u8, out Ticket ticket);    // game thread, 0 bytes, nanoseconds
// any later frame:
if (host.Mailbox.TryReceive(ticket, out var answer)) Apply(answer.Calls);
```

The managed adapters build `await agent.AskAsync(...)` on top of exactly this, with `CancellationToken` mapped to `Cancel`.

### Whistle (speech)

The same shapes, so a designer only learns one library:

```csharp
using var speech = SpeechModel.Open("whistle.cact");
using var ears = speech.Transcriber(in TranscriberOptions.Default);
Utf8Span words = ears.Transcribe(pcm16k);                     // ReadOnlySpan<float> in, UTF-8 out
using var voice = VoiceAgent.Create<Squad>(speech, model);
foreach (var call in voice.Hear(pcm16k)) Apply(call);         // microphone straight to typed calls
```

### Unity

- One package, two plugin builds (net10 and netstandard2.1), as the Unity CI work already does.
- Every core type is blittable or a `ref struct`, so it fits `NativeArray<byte>`, jobs and `ISystem` without conversion.
- The real Unity speed work: today the netstandard2.1 build has no SIMD, because Mono and IL2CPP cannot run `System.Runtime.Intrinsics`. Unity speed therefore needs the hot kernels (int8 GEMV, attention, softmax) compiled through Burst with `Unity.Burst.Intrinsics` (AVX2 and NEON). This is the one change that can move Unity from scalar speed to native speed, and it is where a 4x claim in Unity would come from.

### Optional managed packages

- `Needle.Extensions.AI`: `model.AsChatClient()` implements `IChatClient`. `ChatOptions.Tools` (`AIFunction`) become run-time tools, and calls come back as `FunctionCallContent`. This plugs Needle into Semantic Kernel and every other Microsoft.Extensions.AI consumer.
- `Needle.Hosting`: `services.AddNeedle(o => o.ModelPath = "needle3.cact").AddAgent<Squad>()`.
- `AskAsync`, `StreamAsync` (`IAsyncEnumerable<Token>`), `IAsyncDisposable`.

## 6. Performance

### Targets

- Over 2x faster than upstream 4.x per call on the same machine and thread count, and up to 4x inside Unity.
- These are goals, not measurements. We do not yet know upstream 4.x's speed: the CI job added in this branch measures it first.

### What we know today

- needle-cs is 20 to 35% faster than upstream 3.3.3 on the reference Zen 4 box (measured, issue #57).
- Golden set at 4 threads: 43.9 ms per call, 812 µs per decoded token (measured).
- Upstream 4.x runs the same weights as 3.3.3, so its speed comes from the engine. The first CI run will show where it stands.

### Where the time goes, and the levers

A call is prefill (the query, since the tool list is cached), reasoning tokens, then call tokens. Decode dominates, and each token is dominated by int8 weight streaming. The levers, roughly in order of expected gain:

1. **Generate fewer tokens.** The grammar already forces every byte it can. Typed tools shrink further: an enum is one token, a field name is never sampled. Jump-forward over forced spans is built but changes tokenization, so it ships only if the acceptance suites hold.
2. **Reasoning on demand.** Reasoning is most of the decoded tokens. `Reasoning.Auto` skips it when the confidence head is already sure (the head runs anyway). It ships only if the 4.x acceptance suites stay at 90% or above.
3. **Speculative decoding that is lossless.** Prompt-lookup drafting produces identical tokens. It is off today because small-batch GEMM was too slow, so a dedicated 2 to 4 token verify kernel comes first.
4. **No marshalling at all.** Upstream goes text → JSON → string → your parser. We go token → your struct. This matters most for small calls in Unity, where every managed allocation and P/Invoke counts.
5. **Kernels.** AVX-512 VNNI and NEON dot product paths exist. Next come weight prefetch across layers, fused norm plus quantize, and one barrier per layer instead of three.
6. **Unity: Burst kernels** (above). This is where the 4x would come from.

Every lever is lossless (bit-identical calls) or gated by the acceptance suites. Nothing trades accuracy for speed silently.

### How it is measured

- `bench-4x.yml` (in this branch) runs upstream 4.x (libcactus 4.1.0, verified through its release manifest), upstream 3.3.3 and needle-cs NativeAOT on x64 and arm64 GitHub runners. It interleaves them for at least 5 rounds and reports best-of per metric, at 1, 2 and 4 threads plus the automatic count.
- For claims, we rerun the same script on the self-hosted runner, because shared GitHub runners are noisy.
- For Unity, we play mode-benchmark the scalar build, the Burst build and upstream 4.x through P/Invoke, inside the Unity CI work.
- For accuracy, we port the six upstream acceptance suites (smart home, media player, productivity, wearable, kitchen appliance, data capture; 32 cases each) into `needle eval` and run both engines. A suite passes at 90% or above with no critical failures, the same rule upstream uses.
- Reports always say which numbers are measured and which are inferred.

## 7. Phases

| Phase | What lands | Done when |
|---|---|---|
| 0 (done) | 4.x argument repair and validation, 4.x bench harness and CI job | 19 model-free tests pass, zero allocation; CI produces the 3-engine table |
| 1 | Native arena agent state, `NeedleModel` and `Agent` structs, `Answer` view, `Try` APIs, `Utf8Span` | Existing golden and parity tests pass on the new types; 0 GC objects per agent after construction |
| 2 | Generator v2: `[Tools]` partial struct, call union, streaming decoder, `Text32`-style fields, trigger DFAs, analyzers | Squad sample rewritten with no strings, 0 bytes per call |
| 3 | `Turn` streaming and frame slicing, `AgentHost` and `Mailbox`, Whistle and `VoiceAgent` in the same shape | Burst sample calls Mailbox from a job |
| 4 | 4.x acceptance suites in `needle eval`, then the speed levers one at a time, each A/B tested | Suites at 90% or above; each lever has a measured result, kept or recorded as negative |
| 5 | Optional packages: Extensions.AI, Hosting, async | `IChatClient` conformance tests pass |
| 6 | NuGet packages, a 5-line quickstart, XML docs on every public member, API reference site | `dotnet new needle-agent` works |

## 8. Migration

Breaking changes are intentional. Old types go in one release, and a short table in the README maps each one:

| Today | New |
|---|---|
| `Model` | `NeedleModel` (struct) |
| `NeedleSession` | `Agent` / `Agent<TTools>` (struct) |
| `NeedleOptions` (class) | `AgentOptions` (struct) |
| `CallResult`, `FunctionCall` | `Answer<T>` (ref struct), `CallView` |
| `Reply`, `ReplyStatus`, `ReplyFlags` | `Answer.Kind`, `Validation`, `Failure` |
| `[NeedleTools]` interface + sink | `[Tools]` partial struct |
| `NeedleHost`, `NeedleChannel` | `AgentHost`, `Mailbox` |
| `NeedleService`, `SpeechService` (Task) | `Needle.Extensions` async adapters |

## 9. Risks and honest limits

- **Speed against 4.x is unknown until CI runs.** If upstream 4.x is already much faster than 3.3.3, 2x needs the speculative and reasoning levers to work, and those depend on the acceptance suites.
- **Unity 4x depends on Burst kernels.** Without them, Unity runs scalar code and will be slower than upstream's native library, not faster.
- **Fixed-capacity text** (`Text32` and friends) is the price of zero GC. Text longer than the field is cut by the grammar. The analyzer warns, and `Answer.Text` covers free text.
- **Unity's C# 9** rules out static abstract members and `u8` literals in user code there. The design avoids depending on either; Unity users pass `ReadOnlySpan<char>` or `NativeArray<byte>`.
- **Exceptions are managed objects.** The core never throws on the per-turn path; construction has `Try` variants.
- **A worker thread is an OS thread.** On .NET it is created through `System.Threading.Thread`, the one managed object a host owns, created once. No GC work happens on it per turn.
- **Clean-room rule.** All 4.x behavior is derived from upstream's documentation and its Apache-2.0 Python package, never from the custom-licensed engine source.
