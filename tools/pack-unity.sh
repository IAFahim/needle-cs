#!/usr/bin/env bash
# Builds the two Unity plugin DLLs of com.needle.inference from src/Needle and stamps them with a
# hash of the sources they came from:
#   Runtime/Plugins/netstandard2.1/Needle.Inference.dll      the core every Unity target compiles against
#   Runtime/Plugins/net10.0/Needle.Inference.Net10.dll       the CoreCLR sidecar (Editor, desktop players)
# Both DLLs are committed, so the package installs from a Git URL (issue #56). After changing
# anything under src/Needle, run this script and commit both DLLs.
#
#   tools/pack-unity.sh            rebuild and copy both DLLs (and their .meta files)
#   tools/pack-unity.sh --check    exit 1 when the committed DLLs were not built from the current sources
#   tools/pack-unity.sh --stamp    print the current source stamp
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"

mode="pack"
case "${1:-}" in
  "") ;;
  --check) mode="check" ;;
  --stamp) mode="stamp" ;;
  *) echo "usage: tools/pack-unity.sh [--check | --stamp]" >&2; exit 2 ;;
esac

if [ -n "${NEEDLE_TFM:-}" ]; then
  echo "pack-unity: NEEDLE_TFM is obsolete (both TFMs are always packed); ignoring." >&2
fi

NS21_GUID=9258e34512300c12ea285491e75cd639
NET10_GUID=4fb58f98949287e63de8179319df357b

plugins="unity/com.needle.inference/Runtime/Plugins"
nsdir="$plugins/netstandard2.1"
netdir="$plugins/net10.0"
ns_dll="$nsdir/Needle.Inference.dll"
net_dll="$netdir/Needle.Inference.Net10.dll"

sha256_stdin() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1; else shasum -a 256 | cut -d' ' -f1; fi
}

# The stamp covers every file that compiles into Needle.Inference: src/Needle (minus bin/obj) and
# the repository-wide build props. Carriage returns are dropped before hashing, so a Windows
# checkout with CRLF line endings gets the same stamp as a Linux one.
source_files() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git ls-files --cached --others --exclude-standard -- src/Needle Directory.Build.props Directory.Build.targets
  else
    find src/Needle -type f; ls Directory.Build.props Directory.Build.targets 2>/dev/null || true
  fi | grep -v -E '(^|/)(bin|obj)/' | LC_ALL=C sort -u
}

source_stamp() {
  local f
  source_files | while IFS= read -r f; do
    [ -f "$f" ] || continue
    printf '%s  %s\n' "$(tr -d '\r' < "$f" | sha256_stdin)" "$f"
  done | sha256_stdin
}

stamp="src-sha256:$(source_stamp)"

if [ "$mode" = "stamp" ]; then
  echo "$stamp"
  exit 0
fi

dll_stamp() {
  if [ -f "$1" ]; then LC_ALL=C grep -a -o 'src-sha256:[0-9a-f]\{64\}' "$1" | head -n 1 || true; fi
}

if [ "$mode" = "check" ]; then
  bad=0
  for dll in "$ns_dll" "$net_dll"; do
    found="$(dll_stamp "$dll")"
    if [ ! -f "$dll" ]; then
      echo "pack-unity: $dll is missing." >&2
      bad=1
    elif [ "$found" != "$stamp" ]; then
      echo "pack-unity: $dll is stale (built from ${found:-unstamped sources}, current sources are $stamp)." >&2
      bad=1
    else
      echo "ok $dll ($stamp)"
    fi
  done
  if [ "$bad" != 0 ]; then
    echo "pack-unity: run tools/pack-unity.sh and commit both DLLs under $plugins." >&2
    exit 1
  fi
  exit 0
fi

# Separate artifacts directories keep these builds out of src/Needle/bin and obj, which the test
# project and the CLI use. ContinuousIntegrationBuild maps source paths to /_/ and DebugType=none
# drops the PDB path from the DLL, so the bytes do not depend on where the checkout lives.
build_flags=(-c Release --nologo -v q "-p:NeedleSourceStamp=$stamp" -p:ContinuousIntegrationBuild=true -p:DebugType=none)
dotnet build src/Needle -f netstandard2.1 "${build_flags[@]}" --artifacts-path "$root/out/pack-unity/ns21"
dotnet build src/Needle -f net10.0 "${build_flags[@]}" -p:AssemblyName=Needle.Inference.Net10 --artifacts-path "$root/out/pack-unity/net10"

mkdir -p "$nsdir" "$netdir"

built_ns="$(find "$root/out/pack-unity/ns21/bin" -path '*netstandard2.1*' -name Needle.Inference.dll | head -n 1)"
built_net="$(find "$root/out/pack-unity/net10/bin" -path '*net10.0*' -name Needle.Inference.Net10.dll | head -n 1)"
[ -n "$built_ns" ] || { echo "pack-unity: netstandard2.1 build produced no Needle.Inference.dll" >&2; exit 1; }
[ -n "$built_net" ] || { echo "pack-unity: net10.0 build produced no Needle.Inference.Net10.dll" >&2; exit 1; }
cp "$built_ns" "$ns_dll"
cp "$built_net" "$net_dll"
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

echo "packed $ns_dll (core, $stamp)"
echo "packed $net_dll (CoreCLR sidecar, $stamp)"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && ! git diff --quiet -- "$plugins"; then
  echo "pack-unity: the plugin DLLs changed; commit them with your source change." >&2
fi
