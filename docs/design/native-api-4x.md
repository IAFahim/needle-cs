# needle-cs 4.x: the native API plan

Status: proposal for review. Nothing below past phase 0 is built yet.
Six independent reviewers have read it: section 10 records what they found, and where it differs from sections 3 to 9, section 10 wins.
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
- These are goals, not measurements. The first CI numbers and the restated Unity targets are in section 10.

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

## 10. Council review

Six reviewers read this doc and the code on this branch independently, each with one brief: a .NET API design reviewer, a performance engineer, a Unity game developer, a first-time user, a skeptic and a speech engineer. None of them saw the others' notes. This section is what they found and what we change.

### Where they agree

1. **The hero sample does not compile.** Four of six found this on their own.
   - A nested type and a generated member with the same name (`Squad.Give` and `call.Give`) is CS0102.
   - `record struct` and interpolated string handlers are C# 10, and `u8` is C# 11. Unity is C# 9, the audience the owner cares most about. Worse, `Ask($"...")` still compiles in C# 9 by binding to a `string` overload, so it allocates without a warning.
   - `AgentOptions` holds a `ref struct` (`Utf8Span`) in a plain struct, which is illegal.
   - Section 5 calls `Squad` zero-size; section 3 says a `Squad` value is one call. It cannot be both.
2. **Disposable structs that own native memory are a use-after-free factory.** Copy an `Agent`, dispose one copy, and the other points at freed memory; `default(Agent<T>).Ask()` is an access violation, not an exception. In an IL2CPP player that is a native crash.
3. **`ref struct` views are only lifetime-safe in name.** `var a = agent.Ask(x); agent.Ask(y); a.Calls[0]` reads the second turn's data. `Mailbox.TryReceive` hands out a view the worker is about to overwrite.
4. **The simplest call blocks the caller.** `Ask` takes 50 to 200 ms on desktop .NET and about 700 ms in an IL2CPP player today. In `Update()` that is a visible hitch. Four ways to run a turn (`Ask`, `Turn`, `Mailbox`, async) and none marked "use this one in a game".
5. **Refusals and failures are silent.** `foreach` over a refused answer yields nothing and says nothing.
6. **Parameter descriptions disappeared.** The positional record form has nowhere to put them, and they measurably raise confidence today.
7. **`Reply<TResult>(in TResult)` cannot be implemented** without reflection, because nothing says how an arbitrary struct becomes a tool result.
8. **Fixed-size text needs honesty.** `Text32` counts bytes (about 10 CJK characters), cuts silently, and repair can write past the cap.
9. **What is right, keep it:** compile-time tool binding, decoding straight into structs, the arena per agent, zero allocations per turn as the measured gate, the layering with `IChatClient` as an optional bridge, `Try` twins, the JSON tools path beside the typed one, and breaking the API once with a migration table.

### Where they disagree, and what we pick

| Question | Positions | Decision |
|---|---|---|
| How users declare tools | API design, Unity and skeptic: a `[Tools] partial interface` with methods (C# 9, already supported by the generator). First-time user: a partial struct with nested structs and a checked `Is(out ...)`. | **Interface.** Users write methods; the generator emits the call union and the argument structs inside it, plus checked `Is(out ...)` and an exhaustive dispatcher. Both camps get what they asked for. |
| Owners: struct or class | Skeptic and API design: a sealed class over `SafeHandle` is fine, because one object at startup is not GC pressure. Unity and first-time user: a generation-checked struct handle. | **Generation-checked `readonly struct` handles** (slot plus generation into a native table). It keeps "no managed types" honest and works from Burst, and a stale or default handle fails cleanly: `ObjectDisposedException` on .NET, `NeedleError.Disposed` under Burst, plus `AtomicSafetyHandle` checks in the Unity editor. |
| Attributes for limits | API design: reuse `[Description]` and DataAnnotations `[Range]`. First-time user: `[Range]` and `[Min]` clash with `UnityEngine` (CS0104). | **`[Description]` for tools and parameters** (what Microsoft.Extensions.AI reads), XML doc comments as a fallback, and Needle's own **`[Arg(Min = 1, Max = 99)]`** for limits. |
| Mailbox from Burst jobs | Unity: make it a `NativeQueue`-style container with a `ParallelWriter`. Skeptic: Burst is a C# subset, keep Burst to the kernels. | **Kernels first.** A job-writable mailbox is a phase 5 stretch, not a phase gate. |
| Renames | Skeptic: renaming `Utf8Text32`, `NeedleHost`, `NeedleChannel` is churn. Owner: change the API if it helps. | **Rename once, only where the name improves.** `NeedleModel` and `NeedleAgent` keep the brand on the two types people type first (plain `Agent` clashes with Semantic Kernel). |
| Default confidence floor | First-time user: 0.3, as the README recommends. Parity with 4.x needs upstream's default. | **Upstream's default in the engine**, 0.3 in the Unity agent asset and the samples. |
| Speed targets | Skeptic: drop the 2x and 4x numbers until they are measured. | **Measured now, see below.** Claims quote only the CI table. |

### Correctness the plan must fix first (skeptic, confirmed in code)

- **The typed path skips repair.** `GenerateCallsTyped` never calls `ArgumentRepair`, so the plan's main path would get none of the 4.x validation. Repair moves to a neutral per-call form (tool index plus argument slots) that both the JSON and struct paths feed, and runs as each call closes. A streamed call is marked provisional until the turn's confidence and floor are known.
- **Phase 0 is not done.** Triggers are parsed but never used; `threads = 0` means 1 thread, not automatic; `auto_date` is off where upstream has it on; headless confidence is 1.0 where upstream returns none; and eval scoring must count ungrounded or negated calls as no call.
- **The ungrounded check is too literal.** "7pm", "an hour", "a couple" and "full brightness" are flagged, and a game's inferred coordinates always are. It gets calibrated against upstream's own output on the six suites, and a field can opt out with `[Arg(Inferred = true)]`.
- **Bugs in the repair code on this branch:**
  - A value from one clause can fill a call from another ("set volume to 30 and dim the lights by 10" sets the volume to 10). The whole-request fallback applies only when there is one call.
  - Negation reaches into quoted text ("text Sam that I won't make it" is dropped), and "or" after a negation splits the clause.
  - `start` and `end` are treated as a route's origin and destination, which breaks calendar times.
  - When the history buffer is full, the newest turn is the text that gets dropped.
  - When repaired output outgrows the buffer, the spans are updated but the bytes are not.
  - Smaller: "update" matches the date rule, a field named `number` is treated as a phone number, and month-plus-day never rolls into next year.
- **Bench harness.** The 4.x binding matches the signatures in upstream's Apache-2.0 Python package, so the ABI worry does not hold. Three real gaps remain: upstream's Python sets threads *before* loading, while the bench sets them after, which may be why 4.x shows the same time at every thread count; times are not token-normalised (upstream may decode more tokens); and calls are never checked, so a fast wrong answer would count as a win. All three get fixed before any number is quoted outside this branch.

### Unity additions (Unity reviewer)

- One game path that never computes on the main thread: `pending = agent.Post(line)` returns at once, and `pending.TryTake(out answer)` in a later frame. `AskAsync` through `Awaitable` for those who want it. An analyzer warns when `Ask` is called from `Update`.
- Assets, not file paths: a `.cact` importer producing a model asset, an agent asset (persona, options, examples, fallback lines for refusals), automatic prefix baking on import.
- Inspector wiring with no code: the generator can emit a component with one `UnityEvent` per tool.
- A Playground window: type an order, see calls, confidence, repairs and timing, and save it as a test case.
- Arenas allocated through Unity's allocator so leak detection and the Memory Profiler see them; hosts disposed on domain reload and quit.
- One model per project, one host per world, and requests with a priority, a deadline in frames and cancel-by-owner, so a crowd of NPCs cannot starve the player.
- An honest platform matrix. Today the plugins exclude Android, iOS, WebGL and UWP; consoles need `FromMemory` and core affinity; WebGL has no threads. And the doc must say that Burst kernels already ship but are not active yet.

### Speech additions (speech reviewer)

- Today's streaming re-transcribes the whole window on every pass, which grows quadratically; it becomes incremental.
- `agent.Listen(speech)` instead of a second agent; one shared native ring for microphone audio; `voice.Step(budget, out Hearing e)` with SpeechStarted, Partial, CallReady, Unsure and Done.
- Push-to-talk with pre-roll, voice activity detection, and semantic endpointing: the grammar can see that a call is complete and close the turn after a short silence.
- Command mode: decode speech under a phrase grammar built from the tools, plus `[Spoken]` hotwords for names. This is the largest accuracy gain on offer.
- Word confidence flows into validation, so the game can ask "did you say Lyra?".

### Performance corrections (performance reviewer)

- **The cost model in section 6 is out of date.** The weights are 2-bit and 4-bit codebook, not int8, and the matrix-vector kernel is limited by instruction count, not memory. At 1 thread it is already within about 20% of its floor (7.3 µs measured against about 6.1 µs estimated for a 768x768 layer). So the levers that pay are elsewhere.
- **The biggest lossless lever was missing: decode barely scales with threads.** 1146 µs per token at 1 thread, 812 µs at 4, only 1.41x. The MLP runs on one thread during decode, small serial stages get slower with more threads, and there are about 230 thread barriers per token (not three per layer). Splitting the MLP across threads, running the tiny serial stages on every thread instead of sharding them behind a barrier, and work stealing for every stage is estimated at 30 to 40% less time per call, with bit-identical output.
- **Speculation and jump-forward are weak on x86 desktops.** A multi-token pass costs nearly as much per token as single passes here, so they mostly save fixed per-pass cost, which the item above already cuts. "Typed tools generate fewer tokens" is worth 2 to 4%. Speculation gets calibrated at open and switches on only where it pays (memory-bound phones and ARM with matrix instructions).
- **Cache residency matters on small-L3 machines.** Each token reads 17.5 MB of weights; a 16 MB L3 then misses almost always. Streaming the output head past the cache and keeping each core's rows sticky could give 10 to 25% there. Measure first.
- **Hybrid cores and Unity's own threads cause stragglers.** With hundreds of barriers per token, one slow or preempted core sets the pace. Dynamic chunking everywhere, and 1 to 2 inference threads by default in Unity.
- **Two benchmark tables, not one.** Engine parity (same tokens, same reasoning budget, same schema bytes) and product default against product default, each with the acceptance-suite score next to it. Reasoning is 54% of decoded tokens, so `Reasoning.Auto` is the one lever of 1.5x or more, and it changes outputs.
- **Burst must compile the whole decode step, not only the leaf kernels**, with strict float mode for parity; everything between kernels would otherwise run on Mono. Unity's CoreCLR release gets the .NET 10 SIMD build for free.
- **"No marshalling" is worth microseconds against a 40 ms call.** It prevents frame hitches; it is not a speed lever.
- **Cross-agent batching** (the existing 16-slot batched engine) belongs in the host. It is the throughput lever for many NPCs.
- **Research spike with a kill rule:** a lookup-table kernel for the 2-bit weights on x86 without VNNI, dropped unless the gate layer gets at least 1.3x faster.
- **Unity targets, restated.** 4x is not credible: upstream's native library already runs at native speed inside Unity. The honest Unity targets are engine latency within 1.1x of the .NET 10 build, at most 20 µs of main-thread cost per ask, zero GC, and no p99 frame spikes.

### What we measured since the plan was written

CI run 45745162350, shared 4-vCPU GitHub runners, best of 5 interleaved rounds:

| Machine | Upstream 4.x | needle-cs 1 thread | needle-cs 2 threads | needle-cs 4 threads |
|---|---|---|---|---|
| x64, AMD EPYC Zen 4, golden set | 103 ms | 81 ms (1.27x) | 50 ms (2.05x) | 53 ms (1.92x) |
| x64, game set | 113 ms | 86 ms (1.32x) | 52 ms (2.18x) | 55 ms (2.04x) |
| arm64, Neoverse-N2, golden set | 82 ms | 462 ms (0.18x) | 265 ms (0.31x) | 161 ms (0.51x) |

- On x64 we are about 2x faster than 4.x with 2 or more threads. Prefill is about 3x faster; decode is about 1.6x faster.
- On arm64 we are 2 to 5 times *slower*. This is now the top performance item: the next CI run prints the active CPU tier and a stage profile to find out why.
- Upstream 4.x shows the same time at every thread count. That may be the bench's thread-setting order (above), so the 4.x column is provisional.

### The revised plan

1. **Phase 0, finished properly.** Triggers, automatic threads, `auto_date`, headless confidence, the repair bugs above, and the eval rule. Port the six acceptance suites and score both engines on them. Fix the bench (threads before load, token counts, call checks). Find and fix the arm64 gap.
2. **Phase 1, the core.** Generation-checked handles; versioned views that throw (or return `Failure`) when stale; a caller-buffer overload `Ask(utf8, Span<TCall> calls, out AnswerInfo)` returning `OperationStatus`; options as a plain property struct with `SetSystem` copying into the arena; `answer.Ok` and `answer.Why`; repair on the neutral per-call form for both paths.
3. **Phase 2, the tools.** Evolve today's generator rather than rebuild it: `[Tools] partial interface` with `[Description]` and `[Arg]`; a generated call union with `Kind`, `Is(out ...)` and `Dispatch(ref handler)`; `ref readonly` enumeration; `Text32` exactly 32 bytes, formattable, comparable, with a truncation flag; analyzers for a missing `partial` or `[Tools]`, oversized text and `Ask` in `Update`. Cut the regex-to-DFA compiler (use .NET's own regex, built once) and the "confusable tools" analyzer.
4. **Phase 3, time.** `Post` and `TryTake` as the one game path; per-frame `turn.Run(budget)` returning that slice's events; the same shape for voice. `Continue(resultJson)` replaces `Reply<TResult>`.
5. **Phase 4, speech and Unity.** The speech and Unity additions above.
6. **Phase 5 and 6.** Speed levers one at a time under the acceptance suites, in the performance reviewer's order: thread scaling and fewer barriers first, then prefill tiles, cache residency where it applies, `Reasoning.Auto` as a reported default-versus-default result, and whole-step Burst for Unity. Then the optional packages, optional packages, NuGet with the model as a content package, and a quickstart that a first-time user can run in five minutes.

The five-line quickstart we will hold ourselves to, in C# 9 so it runs in Unity too:

```csharp
[Tools] public partial interface IKitchen { [Description("Set a countdown timer.")] void SetTimer([Description("Length in minutes")] int minutes); }

using var agent = NeedleAgent.Open<IKitchen>("needle3.cact");
foreach (var call in agent.Ask("set a timer for 12 minutes"))
    if (call.Is(out IKitchen.SetTimerArgs t)) Console.WriteLine($"timer: {t.Minutes} min");
```

## 11. What is built (phases 1 to 3)

The quickstart above compiles as written under C# 9 against the netstandard2.1 build, and runs on .NET 10.

| Piece | Type | Notes |
| --- | --- | --- |
| Model | `NeedleModel` (readonly struct handle) | `Open`, `TryOpen`, `FromMemory`, `FromBytes`, `FromStream`. Reference counted: the weights live until the last agent made from them is disposed. |
| Agent | `NeedleAgent` (readonly struct handle) | `model.Agent(json)`, `model.Agent<TTools>()`, `NeedleAgent.Open<TTools>(path)`. A default or disposed handle throws `ObjectDisposedException`. |
| Options | `AgentOptions` (struct, no references) | Every default is upstream 4.x: automatic threads, `auto_date` on, repair on, floor 0.1, 96 reasoning tokens, 2048 context. `Toggle` for three-way switches. |
| Asking | `Ask`, `Continue` | UTF-8, `ReadOnlySpan<char>` and `string` overloads. A caller-buffer `Ask(utf8, Span<byte>, out Reply)` returns `OperationStatus`. |
| Answer | `Answer` (ref struct) | `Kind`, `Ok`, `Why`, `Calls`, `Suppressed`, `Text`, `Thinking`, `Confidence`, `Ungrounded`, `Details`. A view over the agent's buffers: no allocation, and it throws once a later call replaces it. |
| Call | `Call` (ref struct) | `Name`, `Arguments` (`Utf8Span`), `Is("name")`, `Is(out TArgs)`, `TryGetInt32`/`Int64`/`Single`/`Double`/`Boolean`/`Text`/`String`. |
| Tools | `[Tools] partial interface` | Methods and parameters described with `System.ComponentModel.Description`. Generated: `[ToolSchema]`, `XxxArgs` structs, `Calls.Schema`, `Calls.Dispatch(call or answer, handler)`. |
| Game path | `TryPost`, `TryTake`, `IsBusy`, `Take`, `Cancel` | One request in flight per agent, run on the agent's own thread, which sleeps on an event while idle. |
| System facts | `SetSystem` | Copies the facts in and re-primes; the tools stay. |

Not built yet: `[Arg(Min, Max, Inferred)]`, streaming events per frame (`turn.Run(budget)`), the speech additions and the analyzers named in phase 2.
