using System;
using System.IO;
using Gsplat;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

/// <summary>Explicit setup for this standalone diagnostic project, with no XR or game dependency.</summary>
public static class TrackViewerSetup
{
    [MenuItem("Tools/Track3DGS/1. Configure Viewer Rendering")]
    public static void Configure()
    {
        const string folder = "Assets/ViewerSettings";
        Directory.CreateDirectory(folder);
        AssetDatabase.Refresh();
        var renderer = AssetDatabase.LoadAssetAtPath<UniversalRendererData>(folder + "/Renderer.asset");
        if (!renderer)
        {
            renderer = ScriptableObject.CreateInstance<UniversalRendererData>();
            AssetDatabase.CreateAsset(renderer, folder + "/Renderer.asset");
            var type = typeof(GsplatRenderer).Assembly.GetType("Gsplat.GsplatURPFeature", true);
            var feature = (ScriptableRendererFeature)ScriptableObject.CreateInstance(type);
            feature.name = "GsplatURPFeature";
            AssetDatabase.AddObjectToAsset(feature, renderer);
            renderer.rendererFeatures.Add(feature);
            EditorUtility.SetDirty(renderer);
        }
        var pipeline = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(folder + "/Pipeline.asset");
        if (!pipeline)
        {
            pipeline = UniversalRenderPipelineAsset.Create(renderer);
            pipeline.supportsHDR = false;
            pipeline.msaaSampleCount = 1;
            AssetDatabase.CreateAsset(pipeline, folder + "/Pipeline.asset");
        }
        GraphicsSettings.defaultRenderPipeline = pipeline;
        QualitySettings.renderPipeline = pipeline;
        PlayerSettings.colorSpace = ColorSpace.Linear;
        PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64, false);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows64, new[] { GraphicsDeviceType.Direct3D12 });
        GsplatSettings.Instance.EnableGlobalSort = true;
        EditorUtility.SetDirty(GsplatSettings.Instance);
        AssetDatabase.SaveAssets();
        Debug.Log("TRACK_VIEWER_CONFIGURED: URP, Gsplat feature and global sorting. Use D3D12; restart the editor if changing graphics API.");
    }

    // Run in a disposable copy with -batchmode -force-d3d12 -executeMethod TrackViewerSetup.VerifyBatch.
    // No -nographics: these checks exercise real compute shaders and rendered pixels.
    public static void VerifyBatch()
    {
        try
        {
            Configure();
            TrackRoutePreviewBuilder.Rebuild();
            TrackTrainingPreviewWindow.EnsureScene();
            TrackTrainingPreviewWindow.ClearSlots();
            string catalog = TrackTrainingPreviewChecks.CheckSkyCatalog();
            if (catalog.StartsWith("FAIL")) throw new InvalidOperationException(catalog);
            string merge = TrackTrainingPreviewChecks.CheckEditorMerge();
            string comparison = TrackTrainingPreviewChecks.CheckSkyComparison();
            TrackTrainingPreviewWindow.ClearSlots();
            EditorSceneManager.SaveOpenScenes();
            File.WriteAllText("Library/standalone-verification.txt", catalog + "\n" + merge + "\n" + comparison);
            Debug.Log("TRACK_VIEWER_VERIFIED\n" + merge + "\n" + comparison);
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
        catch (Exception error)
        {
            Debug.LogException(error);
            if (Application.isBatchMode) EditorApplication.Exit(1);
            else throw;
        }
    }
}
