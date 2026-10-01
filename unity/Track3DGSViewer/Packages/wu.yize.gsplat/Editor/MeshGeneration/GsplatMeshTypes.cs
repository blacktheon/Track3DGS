using System;
using System.Collections.Generic;
using UnityEngine;

namespace Gsplat.Editor
{
    public struct SplatData
    {
        public Vector3 position;
        public Vector3 normal;
        public Color color;
        public float opacity;
        public Vector3 scale;
    }

    public class MeshData
    {
        public List<Vector3> vertices = new List<Vector3>();
        public List<Vector3> normals = new List<Vector3>();
        public List<int> triangles = new List<int>();
    }
}
