#!/usr/bin/env bash
# Pack the net10.0 Needle.Inference build for the Unity package.
# The DLL is a build artifact (git-ignored); re-run after engine changes.
# GUID pinned: matches the Unity import record for the plugin.
set -euo pipefail
cd "$(dirname "$0")/.."
dotnet build src/Needle -c Release -f net10.0 --nologo -v q
dll="src/Needle/bin/Release/net10.0/Needle.Inference.dll"
dest="unity/com.needle.inference/Runtime/Plugins/net10.0"
mkdir -p "$dest"
cp "$dll" "$dest/Needle.Inference.dll"
printf 'fileFormatVersion: 2\nguid: 9258e34512300c12ea285491e75cd639' > "$dest/Needle.Inference.dll.meta"
echo "packed $dest/Needle.Inference.dll"
