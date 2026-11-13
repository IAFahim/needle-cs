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
