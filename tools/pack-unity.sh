#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${NEEDLE_TFM:-}" ]; then
  echo "pack-unity: NEEDLE_TFM is obsolete (both TFMs are always packed); ignoring." >&2
fi

NS21_GUID=9258e34512300c12ea285491e75cd639
NET10_GUID=4fb58f98949287e63de8179319df357b

dotnet build src/Needle -c Release -f netstandard2.1 --nologo -v q
dotnet build src/Needle -c Release -f net10.0 /p:AssemblyName=Needle.Inference.Net10 --no-incremental --nologo -v q

plugins="unity/com.needle.inference/Runtime/Plugins"
nsdir="$plugins/netstandard2.1"
netdir="$plugins/net10.0"
mkdir -p "$nsdir" "$netdir"

cp "src/Needle/bin/Release/netstandard2.1/Needle.Inference.dll" "$nsdir/Needle.Inference.dll"
cp "src/Needle/bin/Release/net10.0/Needle.Inference.Net10.dll" "$netdir/Needle.Inference.Net10.dll"
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
