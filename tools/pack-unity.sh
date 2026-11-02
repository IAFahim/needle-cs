#!/usr/bin/env bash
# Dual-DLL package layout (N01). Packs BOTH engine targets in one go; no swap,
# no manual step — one checkout builds every player.
#
#   Runtime/Plugins/netstandard2.1/Needle.Inference.dll
#     Assembly "Needle.Inference" — the ONLY compile reference. The asmdef
#     precompiledReferences names this file, and it stays auto-referenced on
#     every platform (Editor, Standalone, Android/IL2CPP, ...). Until N30
#     installs a faster provider, Tiers.Active here is Scalar.
#   Runtime/Plugins/net10.0/Needle.Inference.Net10.dll
#     Assembly "Needle.Inference.Net10" (renamed via /p:AssemblyName so the two
#     can share one process without an identity clash). CoreCLR-only sidecar:
#     compatible with Editor and Standalone desktop players, disabled on every
#     IL2CPP/AOT target (Android, iOS, WebGL, ...). NEVER a compile reference
#     (distinct file name + isExplicitlyReferenced), so the netstandard script
#     profile never sees System.Runtime 10 (V11 CS1705). Loaded at runtime by
#     reflection only — see the N30 hook below; nothing references it.
#
# GUIDs are pinned: the ns2.1 GUID is the long-standing import record (~/Say's
# Library references it; changing it forces a full reimport). The net10 GUID
# was minted once for this layout; never regenerate either.
#
# N30 reflection hook (contract sketch, no code yet):
#   if FrameworkDescription reports .NET 8+ (CoreCLR: Editor, Standalone):
#     asm = Assembly.LoadFrom(
#       <player>/*_Data/Managed/Needle.Inference.Net10.dll   (player)
#       or Plugins/net10.0/Needle.Inference.Net10.dll        (Editor))
#     Kernels.Install(<Net10 provider table from asm>)
#   The ns2.1 core never names Net10 types; a missing file means Scalar,
#   never an error. The sidecar stays out of IL2CPP builds via its .meta, so
#   AOT conversion never sees net10 IL.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${NEEDLE_TFM:-}" ]; then
  echo "pack-unity: NEEDLE_TFM is obsolete (both TFMs are always packed); ignoring." >&2
fi

NS21_GUID=9258e34512300c12ea285491e75cd639
NET10_GUID=4fb58f98949287e63de8179319df357b

dotnet build src/Needle -c Release -f netstandard2.1 --nologo -v q
# --no-incremental: the rename must not reuse the unrenamed build's outputs.
dotnet build src/Needle -c Release -f net10.0 /p:AssemblyName=Needle.Inference.Net10 --no-incremental --nologo -v q

plugins="unity/com.needle.inference/Runtime/Plugins"
nsdir="$plugins/netstandard2.1"
netdir="$plugins/net10.0"
mkdir -p "$nsdir" "$netdir"

cp "src/Needle/bin/Release/netstandard2.1/Needle.Inference.dll" "$nsdir/Needle.Inference.dll"
cp "src/Needle/bin/Release/net10.0/Needle.Inference.Net10.dll" "$netdir/Needle.Inference.Net10.dll"
# One-time migration from the single-GUID layout: a same-name net10 DLL must
# never shadow the ns2.1 compile reference again.
rm -f "$netdir/Needle.Inference.dll" "$netdir/Needle.Inference.dll.meta"

cat > "$nsdir/Needle.Inference.dll.meta" <<EOF
fileFormatVersion: 2
guid: $NS21_GUID
PluginImporter:
  externalObjects: {}
  serializedVersion: 2
  iconMap: {}
  executionOrder: {}
  defineConstraints: []
  isPreloaded: 0
  isOverridable: 1
  isExplicitlyReferenced: 0
  validateReferences: 1
  platformData:
  - first:
      Any:
    second:
      enabled: 1
      settings: {}
  - first:
      Editor: Editor
    second:
      enabled: 0
      settings:
        DefaultValueInitialized: true
  - first:
      Windows Store Apps: WindowsStoreApps
    second:
      enabled: 0
      settings:
        CPU: AnyCPU
  userData:
  assetBundleName:
  assetBundleVariant:
EOF

cat > "$netdir/Needle.Inference.Net10.dll.meta" <<EOF
fileFormatVersion: 2
guid: $NET10_GUID
PluginImporter:
  externalObjects: {}
  serializedVersion: 2
  iconMap: {}
  executionOrder: {}
  defineConstraints: []
  isPreloaded: 0
  isOverridable: 1
  isExplicitlyReferenced: 1
  validateReferences: 1
  platformData:
  - first:
      '': Any
    second:
      enabled: 0
      settings:
        Exclude Android: 1
        Exclude Editor: 0
        Exclude Linux: 0
        Exclude Linux64: 0
        Exclude LinuxUniversal: 0
        Exclude OSXUniversal: 0
        Exclude WebGL: 1
        Exclude Win: 0
        Exclude Win64: 0
        Exclude WindowsStoreApps: 1
        Exclude iOS: 1
        Exclude tvOS: 1
  - first:
      Android: Android
    second:
      enabled: 0
      settings:
        CPU: ARMv7
  - first:
      Any:
    second:
      enabled: 0
      settings: {}
  - first:
      Editor: Editor
    second:
      enabled: 1
      settings:
        CPU: AnyCPU
        DefaultValueInitialized: true
        OS: AnyOS
  - first:
      Facebook: Win
    second:
      enabled: 0
      settings:
        CPU: AnyCPU
  - first:
      Facebook: Win64
    second:
      enabled: 0
      settings:
        CPU: AnyCPU
  - first:
      Standalone: Linux
    second:
      enabled: 1
      settings:
        CPU: x86
  - first:
      Standalone: Linux64
    second:
      enabled: 1
      settings:
        CPU: AnyCPU
  - first:
      Standalone: LinuxUniversal
    second:
      enabled: 1
      settings: {}
  - first:
      Standalone: OSXUniversal
    second:
      enabled: 1
      settings:
        CPU: AnyCPU
  - first:
      Standalone: Win
    second:
      enabled: 1
      settings:
        CPU: AnyCPU
  - first:
      Standalone: Win64
    second:
      enabled: 1
      settings:
        CPU: AnyCPU
  - first:
      WebGL: WebGL
    second:
      enabled: 0
      settings: {}
  - first:
      Windows Store Apps: WindowsStoreApps
    second:
      enabled: 0
      settings:
        CPU: AnyCPU
        DontProcess: false
        PlaceholderPath:
        SDK: AnySDK
        ScriptingBackend: AnyScriptingBackend
  - first:
      iPhone: iOS
    second:
      enabled: 0
      settings:
        AddToEmbeddedBinaries: false
        CompileFlags:
        FrameworkDependencies:
  - first:
      tvOS: tvOS
    second:
      enabled: 0
      settings:
        CompileFlags:
        FrameworkDependencies:
  userData:
  assetBundleName:
  assetBundleVariant:
EOF

echo "packed $nsdir/Needle.Inference.dll (ref)"
echo "packed $netdir/Needle.Inference.Net10.dll (CoreCLR sidecar)"
