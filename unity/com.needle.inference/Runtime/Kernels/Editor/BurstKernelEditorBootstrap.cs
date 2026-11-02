// N31: EditMode bootstrap for the Burst provider.
//
// The runtime assembly cannot name UnityEditor types (it ships in players),
// so the [InitializeOnLoadMethod] half of the installer lives here in an
// Editor-only assembly. Guarded: a failed compile must never break script
// reload; inference falls back to Scalar and the failure is visible once.
using System;
using UnityEditor;
using UnityEngine;

namespace Needle.BurstKernels
{
    public static class BurstKernelEditorBootstrap
    {
        [InitializeOnLoadMethod]
        static void InstallForEditMode()
        {
            try
            {
                BurstKernelInstaller.EnsureInstalled();
                Debug.Log("[needle] " + BurstKernelInstaller.Describe());
            }
            catch (Exception e)
            {
                Debug.LogWarning("[needle] Burst kernel install failed, using Scalar: " + e.Message);
            }
        }
    }
}
