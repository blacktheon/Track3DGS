using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEditor;

namespace Gsplat.Editor
{
    public static class MeshDecimator
    {

        public static MeshData SimplifyMeshData(MeshData inputData, int targetTris)
        {
            int currentTris = inputData.triangles.Count / 3;
            if (currentTris <= targetTris) return inputData;

            List<Vector3> verts = inputData.vertices;
            List<int> tris = inputData.triangles;

            Bounds bounds = new Bounds(verts[0], Vector3.zero);
            foreach (var v in verts) bounds.Encapsulate(v);

            // Safety limit to prevent the mesh from collapsing into a single point
            int subdivisions = Mathf.Max(16, Mathf.RoundToInt(Mathf.Sqrt(targetTris)));
            float gridCellSize = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z) / subdivisions;

            Dictionary<Vector3Int, int> vertexClusterMap = new Dictionary<Vector3Int, int>();
            MeshData simplified = new MeshData();
            int[] remappedIndex = new int[verts.Count];

            for (int i = 0; i < verts.Count; i++)
            {
                Vector3Int cell = new Vector3Int(
                    Mathf.FloorToInt((verts[i].x - bounds.min.x) / gridCellSize),
                    Mathf.FloorToInt((verts[i].y - bounds.min.y) / gridCellSize),
                    Mathf.FloorToInt((verts[i].z - bounds.min.z) / gridCellSize)
                );

                if (!vertexClusterMap.TryGetValue(cell, out int clusterIdx))
                {
                    clusterIdx = simplified.vertices.Count;
                    vertexClusterMap[cell] = clusterIdx;
                    simplified.vertices.Add(verts[i]);
                }

                remappedIndex[i] = clusterIdx;
            }

            for (int i = 0; i < tris.Count; i += 3)
            {
                int i0 = remappedIndex[tris[i]];
                int i1 = remappedIndex[tris[i + 1]];
                int i2 = remappedIndex[tris[i + 2]];

                if (i0 != i1 && i1 != i2 && i0 != i2)
                {
                    simplified.triangles.Add(i0);
                    simplified.triangles.Add(i1);
                    simplified.triangles.Add(i2);
                }
            }

            return simplified;
        }

    }
}
