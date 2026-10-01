using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEditor;

namespace Gsplat.Editor
{
    public static class GsplatMeshBuilder
    {

        public static Mesh GenerateMeshFromPly(string plyFilePath, int resolution, float threshold,
            bool enablePointFiltering, float minOpacity, float maxScale, SourceCoordinates sourceCoordinates,
            bool enableOutlier, float outlierRadius, int minNeighbors,
            bool enableSimplification, int targetTris, bool generateNormals = true, ProgressCallback progressCallback = null)
        {
            if (!File.Exists(plyFilePath))
            {
                Debug.LogError($"[Gsplat Mesh Generator] File not found: {plyFilePath}");
                return null;
            }

            progressCallback?.Invoke("Generating Mesh: Parsing PLY...", 0.1f);
            float effectiveMinOpacity = enablePointFiltering ? minOpacity : 0.0f;
            float effectiveMaxScale = enablePointFiltering ? maxScale : float.MaxValue;
            bool effectiveOutlier = enablePointFiltering && enableOutlier;

            List<SplatData> splats = PlyParser.ParsePlyFileInternal(plyFilePath, effectiveMinOpacity, effectiveMaxScale, sourceCoordinates);
            if (effectiveOutlier && splats != null && splats.Count > 0)
            {
                splats = PlyParser.FilterIsolatedOutliers(splats, outlierRadius, minNeighbors);
            }

            if (splats == null || splats.Count == 0)
            {
                Debug.LogWarning($"[Gsplat Mesh Generator] No valid points found in {plyFilePath}.");
                return null;
            }

            progressCallback?.Invoke($"Generating Mesh: Reconstructing ({splats.Count:N0} points)...", 0.4f);
            MeshData rawMeshData = VoxelReconstructor.ComputeMesh(splats, resolution, threshold);

            if (rawMeshData == null || rawMeshData.vertices.Count == 0)
            {
                Debug.LogWarning($"[Gsplat Mesh Generator] Reconstruction failed to produce vertices for {plyFilePath}.");
                return null;
            }

            if (enableSimplification && rawMeshData.triangles.Count / 3 > targetTris)
            {
                progressCallback?.Invoke("Generating Mesh: Simplifying Mesh...", 0.7f);
                rawMeshData = MeshDecimator.SimplifyMeshData(rawMeshData, targetTris);
            }

            progressCallback?.Invoke("Generating Mesh: Finalizing Unity Mesh...", 0.9f);
            Mesh rawMesh = new Mesh();
            rawMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            rawMesh.SetVertices(rawMeshData.vertices);
            rawMesh.SetTriangles(rawMeshData.triangles, 0);
            if (generateNormals)
            {
                rawMesh.RecalculateNormals();
            }
            rawMesh.RecalculateBounds();

            return rawMesh;
        }

    }
}
