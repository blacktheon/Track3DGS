using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEditor;

namespace Gsplat.Editor
{
    public static class MeshExporter
    {

        public static void ExportToObj(Mesh mesh, string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("# Gsplat Mesh Generator OBJ Export");
                writer.WriteLine($"# Vertices: {mesh.vertexCount}");

                Vector3[] verts = mesh.vertices;
                Vector3[] normals = mesh.normals;

                for (int i = 0; i < verts.Length; i++)
                {
                    Vector3 v = verts[i];
                    writer.WriteLine($"v {-v.x:F6} {v.y:F6} {v.z:F6}");
                }

                foreach (Vector3 n in normals)
                {
                    writer.WriteLine($"vn {-n.x:F6} {n.y:F6} {n.z:F6}");
                }

                bool hasNormals = normals != null && normals.Length > 0;
                int[] tris = mesh.triangles;
                for (int i = 0; i < tris.Length; i += 3)
                {
                    int idx1 = tris[i] + 1;
                    int idx2 = tris[i + 1] + 1;
                    int idx3 = tris[i + 2] + 1;
                    if (hasNormals)
                    {
                        writer.WriteLine($"f {idx1}//{idx1} {idx3}//{idx3} {idx2}//{idx2}");
                    }
                    else
                    {
                        writer.WriteLine($"f {idx1} {idx3} {idx2}");
                    }
                }
            }
        }

    }
}
