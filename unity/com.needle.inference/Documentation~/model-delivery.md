# Model delivery per platform

One call loads a model bundled as `StreamingAssets/<file>` on every player.
Start the coroutine from a loading scene: first open faults pages in and
blocks for a large-model read.

```csharp
IEnumerator Start()
{
    Model model = null;
    string error = null;
    yield return ModelLoader.OpenAsync(
        "needle3.cact",
        onReady: m => model = m,
        onError: e => error = e);
    if (error != null) { Debug.LogError(error); yield break; }
    // ... build NeedleSession / Whistle around model, Dispose on teardown
}
```

`ModelLoader.CanOpenDirect("needle3.cact")` tells UI code whether this is a
fast direct open or an Android download (show progress accordingly).

## What happens where

| Platform | `streamingAssetsPath` | Strategy |
|---|---|---|
| Editor | real directory | `Model.Open` (mmap on net10, read fallback on ns2.1) |
| Windows/macOS/Linux player | real directory | `Model.Open` directly |
| iOS player | app bundle `Data/Raw` | `Model.Open` directly |
| Android player | `jar:file://…apk!/assets` | `UnityWebRequest` out of the APK, then open |

`Model.Open` on a file path is the whole Linux-player path, and it stays the
right call as mapping improves underneath (N22 mmap, W05 large pages): no
caller changes when the core gets faster.

## Android: cache or hold in memory

`OpenAsync` defaults to `cache: true`: the download is written once to
`Application.persistentDataPath` (atomically, via a `.tmp` rename) and every
launch — including the first rerun after download — opens that copy with
`Model.Open`, so RSS and open time match desktop. Pass `cache: false` to hold
the model with `Model.FromBytes` instead: no storage footprint, but every
launch pays the download plus a full managed-to-native copy (~2x transient
memory while loading).

Pass `expectedBytes` (your `.cact` length) to reject truncated downloads and
stale caches; a length mismatch deletes the cache and re-downloads. Renaming
the bundled file (e.g. `needle3-v2.cact`) has the same effect and is the
recommended pattern when you ship a new model — the old cache is just left
behind. There is no re-download check beyond length: same name plus same
length means the cache wins.

## Migrating the samples

`SquadCommander`, `NpcSample` and `VoiceSample` currently guard
`Path.Combine(Application.streamingAssetsPath, ModelFile)` with `File.Exists`
and call `Model.Open`. That check fails inside an APK. Replace both lines
with one `OpenAsync` coroutine; keep everything after the open unchanged.
`VoiceSample`'s clip (`test.wav`) has the same APK problem — fetch it with
plain `UnityWebRequest` (or `UnityWebRequestMultimedia.GetAudioClip`) since
`Wav.Read` also needs a real file.

## Out of scope

Per-device build settings, Burst AOT flags and the conformance matrix
(Android arm64 with and without dotprod, iOS, Windows/Linux IL2CPP x64) are
N41's device matrix, not this loader. WebGL has no file system or threads
here and is out of scope until the IL2CPP rows are green.
