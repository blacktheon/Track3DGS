using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace Gsplat.Editor
{
    public enum ExportFormat
    {
        UnityMeshAsset,
        ObjFile
    }

    public class GsplatMeshGenerator : EditorWindow
    {
        private CancellationTokenSource cts;

        private void OnDestroy()
        {
            if (cts != null)
            {
                cts.Cancel();
                cts.Dispose();
                cts = null;
            }
        }

        [MenuItem("Tools/Gsplat/Mesh Generator")]
        public static void ShowWindow()
        {
            var window = GetWindow<GsplatMeshGenerator>("Gsplat Mesh Generator");
            window.minSize = new Vector2(420, 560);
            window.isProcessing = false;
            window.Show();
            window.Focus();
        }

        public static void ShowWindowWithFile(string assetPath)
        {
            var window = GetWindow<GsplatMeshGenerator>("Gsplat Mesh Generator");
            window.minSize = new Vector2(420, 560);
            window.isProcessing = false;
            
            if (!string.IsNullOrEmpty(assetPath) && assetPath.EndsWith(".ply", StringComparison.OrdinalIgnoreCase))
            {
                window.plyFilePath = Path.Combine(Directory.GetCurrentDirectory(), assetPath);
            }
            
            window.Show();
            window.Focus();
        }

        private string plyFilePath = "";
        private ExportFormat exportFormat = ExportFormat.UnityMeshAsset;

        // Filtering Parameters for 3D Gaussians / Point Clouds
        private bool enablePointFiltering = true;
        private float minOpacityThreshold = 0.15f;
        private float maxScaleCutoff = 10.0f;
        private SourceCoordinates sourceCoordinates = SourceCoordinates.RUB;
        private bool enableOutlierFilter = true;
        private float outlierSearchRadius = 0.08f;
        private int minNeighborsCount = 8;

        // Voxel / Reconstruction parameters
        private int voxelGridResolution = 96;
        private float isoSurfaceThreshold = 0.1f;

        // Optimization parameters
        private bool enableSimplification = true;
        private int targetTriangleCount = 50000;
        private bool autoInstantiateInScene = true;
        private bool generateNormals = true;

        private Vector2 scrollPos;
        private bool isProcessing = false;
        private float progress = 0.0f;
        private string statusMessage = "Ready";

        private void OnGUI()
        {
            EditorGUIUtility.labelWidth = 200f;
            
            EditorGUILayout.Space(10);
            GUILayout.Label("Gsplat Mesh Generator", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Convert 3D Gaussian Splats or Point Clouds (.ply) to Unity 3D Meshes", EditorStyles.miniLabel);
            EditorGUILayout.Space(10);

            scrollPos = EditorGUILayout.BeginScrollView(scrollPos);

            // --- File Selection ---
            EditorGUILayout.BeginVertical("box");
            GUILayout.Label("Source File", EditorStyles.boldLabel);
            EditorGUILayout.BeginHorizontal();
            plyFilePath = EditorGUILayout.TextField(".ply File Path", plyFilePath);
            if (GUILayout.Button("Browse", GUILayout.Width(70)))
            {
                string path = EditorUtility.OpenFilePanel("Select .ply File", Application.dataPath, "ply");
                if (!string.IsNullOrEmpty(path))
                {
                    plyFilePath = path;
                }
            }
            EditorGUILayout.EndHorizontal();

            // Quick select buttons for files in project
            if (GUILayout.Button("Use Selected Asset in Project"))
            {
                if (Selection.activeObject != null)
                {
                    string assetPath = AssetDatabase.GetAssetPath(Selection.activeObject);
                    if (assetPath.EndsWith(".ply", StringComparison.OrdinalIgnoreCase))
                    {
                        plyFilePath = Path.Combine(Directory.GetCurrentDirectory(), assetPath);
                    }
                    else
                    {
                        EditorUtility.DisplayDialog("Invalid File", "Selected asset is not a .ply file.", "OK");
                    }
                }
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // --- Export Settings ---
            EditorGUILayout.BeginVertical("box");
            GUILayout.Label("Generation Settings", EditorStyles.boldLabel);
            sourceCoordinates = (SourceCoordinates)EditorGUILayout.EnumPopup(new GUIContent("Source Coordinates", "The original coordinate system of the .ply file. Use RUB for standard gaussian splatting outputs."), sourceCoordinates);
            exportFormat = (ExportFormat)EditorGUILayout.EnumPopup(new GUIContent("Export Format", "Save as a native Unity Mesh Asset or a generic OBJ file."), exportFormat);
            generateNormals = EditorGUILayout.Toggle(new GUIContent("Generate Normals", "Calculates vertex normals for smooth shading. Disable for a smaller file size if using purely for invisible collisions."), generateNormals);
            autoInstantiateInScene = EditorGUILayout.Toggle(new GUIContent("Instantiate in Scene", "Automatically spawn the generated mesh into the active scene with a MeshCollider attached."), autoInstantiateInScene);
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // --- Gaussian / Point Cloud Pre-filtering ---
            EditorGUILayout.BeginVertical("box");
            GUILayout.Label("Splat & Point Filtering", EditorStyles.boldLabel);
            enablePointFiltering = EditorGUILayout.Toggle(new GUIContent("Enable Point Filtering", "Filter out low quality or massive splats before building the mesh to improve speed and quality."), enablePointFiltering);
            if (enablePointFiltering)
            {
                minOpacityThreshold = EditorGUILayout.Slider(new GUIContent("Min Opacity (Filter Floaters)", "Ignores transparent, ghost-like splats. Higher values produce cleaner meshes but may erode thin surfaces."), minOpacityThreshold, 0.0f, 1.0f);
                maxScaleCutoff = EditorGUILayout.FloatField(new GUIContent("Max Scale Cutoff", "Ignores massively stretched splats that usually represent background sky or noise."), maxScaleCutoff);
                enableOutlierFilter = EditorGUILayout.Toggle(new GUIContent("Enable Outlier Search Filter", "Removes isolated floating splats that have very few neighbors."), enableOutlierFilter);
                if (enableOutlierFilter)
                {
                    outlierSearchRadius = EditorGUILayout.Slider(new GUIContent("Outlier Search Radius", "The physical distance to search for neighboring splats."), outlierSearchRadius, 0.01f, 20.0f);
                    minNeighborsCount = EditorGUILayout.IntSlider(new GUIContent("Min Neighbor Density", "The minimum number of neighbors required within the search radius for a splat to survive."), minNeighborsCount, 2, 500);
                }
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // --- Reconstruction Specific Options ---
            EditorGUILayout.BeginVertical("box");
            GUILayout.Label("Reconstruction Parameters", EditorStyles.boldLabel);
            voxelGridResolution = EditorGUILayout.IntSlider(new GUIContent("Voxel Grid Resolution", "The density of the marching cubes grid. Higher values take longer but capture finer details."), voxelGridResolution, 32, 1024);
            isoSurfaceThreshold = EditorGUILayout.Slider(new GUIContent("Voxel Opacity Cutoff", "The accumulated density required for a voxel to be considered solid inside the mesh."), isoSurfaceThreshold, 0.01f, 1.0f);
            EditorGUILayout.EndVertical();

            EditorGUILayout.Space(10);

            // --- Optimization & Decimation ---
            EditorGUILayout.BeginVertical("box");
            GUILayout.Label("Optimization & Decimation", EditorStyles.boldLabel);
            enableSimplification = EditorGUILayout.Toggle(new GUIContent("Enable Decimation", "Aggressively reduce the triangle count of the final mesh using extremely fast vertex clustering."), enableSimplification);
            if (enableSimplification)
            {
                targetTriangleCount = EditorGUILayout.IntField(new GUIContent("Target Triangle Count", "Approximate target number of triangles. The algorithm will adjust spatial clustering to roughly hit this goal."), targetTriangleCount);
            }
            EditorGUILayout.EndVertical();

            EditorGUILayout.EndScrollView();

            EditorGUILayout.Space(10);

            // --- Action Button ---
            EditorGUI.BeginDisabledGroup(isProcessing || string.IsNullOrEmpty(plyFilePath));
            if (GUILayout.Button("Generate Mesh", GUILayout.Height(24)))
            {
                GenerateMeshProcessAsync();
            }
            EditorGUI.EndDisabledGroup();

            if (isProcessing)
            {
                EditorGUILayout.HelpBox($"Processing: {statusMessage}", MessageType.Info);
            }
        }

        [MenuItem("Assets/Generate Mesh from .ply", false, 30)]
        private static void ContextGenerateMesh()
        {
            var sel = Selection.activeObject;
            if (sel != null)
            {
                string path = AssetDatabase.GetAssetPath(sel);
                if (path.EndsWith(".ply", StringComparison.OrdinalIgnoreCase))
                {
                    var window = GetWindow<GsplatMeshGenerator>("Gsplat Mesh Generator");
                    window.minSize = new Vector2(420, 560);
                    window.plyFilePath = Path.Combine(Directory.GetCurrentDirectory(), path);
                    window.isProcessing = false;
                    window.Show();
                    window.Focus();
                }
            }
        }

        [MenuItem("Assets/Generate Mesh from .ply", true)]
        private static bool ValidateContextGenerateMesh()
        {
            var sel = Selection.activeObject;
            if (sel == null) return false;
            string path = AssetDatabase.GetAssetPath(sel);
            return path.EndsWith(".ply", StringComparison.OrdinalIgnoreCase);
        }

        private async void GenerateMeshProcessAsync()
        {
            if (!File.Exists(plyFilePath))
            {
                EditorUtility.DisplayDialog("Error", ".ply file does not exist at specified path.", "OK");
                return;
            }

            if (cts != null)
            {
                cts.Cancel();
                cts.Dispose();
            }
            cts = new CancellationTokenSource();
            var token = cts.Token;

            isProcessing = true;
            progress = 0.05f;
            statusMessage = "Parsing .ply Header & Points...";
            EditorUtility.DisplayCancelableProgressBar("Gsplat Mesh Generator", statusMessage, progress);
            Repaint();

            try
            {
                // 1. Parse PLY Data on Background Thread
                string localFilePath = plyFilePath;
                float localMinOpacity = enablePointFiltering ? minOpacityThreshold : 0.0f;
                float localMaxScale = enablePointFiltering ? maxScaleCutoff : float.MaxValue;
                SourceCoordinates localSourceCoords = sourceCoordinates;
                bool localEnableOutlier = enablePointFiltering && enableOutlierFilter;
                float localOutlierRadius = outlierSearchRadius;
                int localMinNeighbors = minNeighborsCount;

                List<SplatData> splats = await Task.Run(() => {
                    var parsed = PlyParser.ParsePlyFileInternal(localFilePath, localMinOpacity, localMaxScale, localSourceCoords);
                    if (localEnableOutlier && parsed != null && parsed.Count > 0)
                    {
                        parsed = PlyParser.FilterIsolatedOutliers(parsed, localOutlierRadius, localMinNeighbors);
                    }
                    return parsed;
                }, token);
                
                token.ThrowIfCancellationRequested();

                if (splats == null || splats.Count == 0)
                {
                    EditorUtility.DisplayDialog("Error", "Failed to parse .ply points or no points passed opacity filter.", "OK");
                    return;
                }

                progress = 0.35f;
                statusMessage = $"Reconstructing surface ({splats.Count:N0} valid points)...";
                EditorUtility.DisplayCancelableProgressBar("Gsplat Mesh Generator", statusMessage, progress);
                Repaint();

                // 2. Reconstruct Mesh Data on Background Thread
                int localRes = voxelGridResolution;
                float localIso = isoSurfaceThreshold;

                MeshData rawMeshData = await Task.Run(() => VoxelReconstructor.ComputeMesh(splats, localRes, localIso), token);

                token.ThrowIfCancellationRequested();

                if (rawMeshData == null || rawMeshData.vertices.Count == 0)
                {
                    EditorUtility.DisplayDialog("Error", "Surface reconstruction failed to produce vertices.", "OK");
                    return;
                }

                progress = 0.70f;
                statusMessage = "Optimizing and simplifying mesh...";
                EditorUtility.DisplayCancelableProgressBar("Gsplat Mesh Generator", statusMessage, progress);
                Repaint();

                // 3. Mesh Simplification on Background Thread
                bool localSimplification = enableSimplification;
                int localTargetTris = targetTriangleCount;

                if (localSimplification && rawMeshData.triangles.Count / 3 > localTargetTris)
                {
                    rawMeshData = await Task.Run(() => MeshDecimator.SimplifyMeshData(rawMeshData, localTargetTris), token);
                }
                
                token.ThrowIfCancellationRequested();

                if (rawMeshData.vertices.Count < 3 || rawMeshData.triangles.Count < 3)
                {
                    EditorUtility.DisplayDialog("Error", "Mesh generation failed or was completely decimated into nothing. No object will be created.", "OK");
                    return;
                }

                progress = 0.90f;
                statusMessage = "Saving Asset to Unity Project...";
                EditorUtility.DisplayCancelableProgressBar("Gsplat Mesh Generator", statusMessage, progress);
                Repaint();

                // 4. Create Unity Mesh on Main Thread
                Mesh rawMesh = new Mesh();
                rawMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
                rawMesh.SetVertices(rawMeshData.vertices);
                rawMesh.SetTriangles(rawMeshData.triangles, 0);
                if (generateNormals)
                {
                    rawMesh.RecalculateNormals();
                }
                rawMesh.RecalculateBounds();

                // 5. Save Asset / OBJ
                string outputDirectory = "Assets/GeneratedMeshes";
                if (!Directory.Exists(outputDirectory))
                {
                    Directory.CreateDirectory(outputDirectory);
                }

                string fileName = Path.GetFileNameWithoutExtension(plyFilePath) + "_Mesh";
                string assetPath = $"{outputDirectory}/{fileName}.asset";

                if (exportFormat == ExportFormat.UnityMeshAsset)
                {
                    AssetDatabase.CreateAsset(rawMesh, assetPath);
                    AssetDatabase.SaveAssets();
                    AssetDatabase.Refresh();

                    Debug.Log($"[Gsplat Mesh Generator] Mesh saved to {assetPath} ({rawMesh.vertexCount:N0} vertices, {rawMesh.triangles.Length / 3:N0} triangles).");

                    if (autoInstantiateInScene)
                    {
                        InstantiateMeshInScene(rawMesh, fileName, assetPath);
                    }
                }
                else if (exportFormat == ExportFormat.ObjFile)
                {
                    string objPath = $"{outputDirectory}/{fileName}.obj";
                    MeshExporter.ExportToObj(rawMesh, objPath);
                    
                    // Force Unity to import the .obj immediately so it exists on disk
                    AssetDatabase.ImportAsset(objPath, ImportAssetOptions.ForceSynchronousImport);
                    
                    Debug.Log($"[Gsplat Mesh Generator] OBJ exported to {objPath}");

                    if (autoInstantiateInScene)
                    {
                        InstantiateMeshInScene(rawMesh, fileName, objPath);
                    }
                }

                progress = 1.0f;
                // Removed success dialog so it doesn't interrupt workflow
            }
            catch (OperationCanceledException)
            {
                Debug.Log("[Gsplat Mesh Generator] Generation was canceled.");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Gsplat Mesh Generator] Error generating mesh: {ex}");
                EditorUtility.DisplayDialog("Error", $"Generation failed: {ex.Message}", "OK");
            }
            finally
            {
                isProcessing = false;
                EditorUtility.ClearProgressBar();
                Repaint();
            }
        }

        private void InstantiateMeshInScene(Mesh mesh, string name, string meshAssetPath)
        {
            GameObject go = new GameObject(name);
            MeshCollider mc = go.AddComponent<MeshCollider>();

            Mesh loadedMesh = AssetDatabase.LoadAssetAtPath<Mesh>(meshAssetPath);
            
            // If it's an OBJ, LoadAssetAtPath<Mesh> might fail because the main asset is a GameObject.
            // We have to extract the sub-asset Mesh instead.
            if (loadedMesh == null && meshAssetPath.EndsWith(".obj", StringComparison.OrdinalIgnoreCase))
            {
                var allAssets = AssetDatabase.LoadAllAssetsAtPath(meshAssetPath);
                foreach (var asset in allAssets)
                {
                    if (asset is Mesh m)
                    {
                        loadedMesh = m;
                        break;
                    }
                }
            }

            mc.sharedMesh = loadedMesh != null ? loadedMesh : mesh;

            Selection.activeGameObject = go;
            EditorGUIUtility.PingObject(go);
        }


    }
}
