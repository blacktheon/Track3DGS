using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEditor;

namespace Gsplat.Editor
{
    public static class PlyParser
    {


        public static List<SplatData> ParsePlyFileInternal(string filePath, float minOpacity, float maxScale, SourceCoordinates sourceCoordinates)
        {
            List<SplatData> result = new List<SplatData>();
            (float xSign, float ySign, float zSign) = GsplatUtils.AxisSigns(sourceCoordinates);

            using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (BinaryReader br = new BinaryReader(fs))
            {
                // Read ASCII Header
                int vertexCount = 0;
                List<PlyProperty> properties = new List<PlyProperty>();
                bool isHeaderFinished = false;

                while (!isHeaderFinished && fs.Position < fs.Length)
                {
                    string line = ReadAsciiLine(br).Trim();
                    if (line.StartsWith("element vertex"))
                    {
                        string[] parts = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 3)
                        {
                            int.TryParse(parts[2], out vertexCount);
                        }
                    }
                    else if (line.StartsWith("property"))
                    {
                        string[] parts = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 3)
                        {
                            PlyProperty prop = new PlyProperty
                            {
                                type = parts[1],
                                name = parts[2]
                            };
                            properties.Add(prop);
                        }
                    }
                    else if (line == "end_header")
                    {
                        isHeaderFinished = true;
                    }
                }

                if (!isHeaderFinished || vertexCount == 0)
                {
                    Debug.LogError("Invalid PLY header or 0 vertices found.");
                    return null;
                }

                // Property Indices / Offsets
                int xIndex = properties.FindIndex(p => p.name == "x");
                int yIndex = properties.FindIndex(p => p.name == "y");
                int zIndex = properties.FindIndex(p => p.name == "z");

                int nxIndex = properties.FindIndex(p => p.name == "nx");
                int nyIndex = properties.FindIndex(p => p.name == "ny");
                int nzIndex = properties.FindIndex(p => p.name == "nz");

                int opacityIndex = properties.FindIndex(p => p.name == "opacity");
                int fdc0Index = properties.FindIndex(p => p.name == "f_dc_0");
                int fdc1Index = properties.FindIndex(p => p.name == "f_dc_1");
                int fdc2Index = properties.FindIndex(p => p.name == "f_dc_2");

                int rIndex = properties.FindIndex(p => p.name == "red" || p.name == "r");
                int gIndex = properties.FindIndex(p => p.name == "green" || p.name == "g");
                int bIndex = properties.FindIndex(p => p.name == "blue" || p.name == "b");

                int scale0Index = properties.FindIndex(p => p.name == "scale_0");
                int scale1Index = properties.FindIndex(p => p.name == "scale_1");
                int scale2Index = properties.FindIndex(p => p.name == "scale_2");

                const float SH_C0 = 0.28209479177387814f;

                // Read Binary Vertices
                for (int i = 0; i < vertexCount; i++)
                {
                    Vector3 pos = Vector3.zero;
                    Vector3 norm = Vector3.up;
                    Color col = Color.white;
                    float rawOpacity = 0.0f;
                    Vector3 scale = Vector3.one;

                    for (int p = 0; p < properties.Count; p++)
                    {
                        float val = ReadPropertyValue(br, properties[p].type);

                        if (p == xIndex) pos.x = val;
                        else if (p == yIndex) pos.y = val;
                        else if (p == zIndex) pos.z = val;
                        else if (p == nxIndex) norm.x = val;
                        else if (p == nyIndex) norm.y = val;
                        else if (p == nzIndex) norm.z = val;
                        else if (p == opacityIndex) rawOpacity = val;
                        else if (p == scale0Index) scale.x = Mathf.Exp(val);
                        else if (p == scale1Index) scale.y = Mathf.Exp(val);
                        else if (p == scale2Index) scale.z = Mathf.Exp(val);
                        else if (p == fdc0Index) col.r = Mathf.Clamp01(0.5f + SH_C0 * val);
                        else if (p == fdc1Index) col.g = Mathf.Clamp01(0.5f + SH_C0 * val);
                        else if (p == fdc2Index) col.b = Mathf.Clamp01(0.5f + SH_C0 * val);
                        else if (p == rIndex) col.r = val > 1.0f ? val / 255.0f : val;
                        else if (p == gIndex) col.g = val > 1.0f ? val / 255.0f : val;
                        else if (p == bIndex) col.b = val > 1.0f ? val / 255.0f : val;
                    }

                    // Opacity transformation for 3D Gaussian Splats: sigmoid(opacity)
                    float realOpacity = opacityIndex >= 0 ? 1.0f / (1.0f + Mathf.Exp(-rawOpacity)) : 1.0f;

                    // Filtering floaters & oversized background splats
                    if (realOpacity < minOpacity) continue;
                    if (scale.x > maxScale || scale.y > maxScale || scale.z > maxScale) continue;

                    // Coordinate system transformation
                    pos.x *= xSign; pos.y *= ySign; pos.z *= zSign;
                    norm.x *= xSign; norm.y *= ySign; norm.z *= zSign;

                    result.Add(new SplatData
                    {
                        position = pos,
                        normal = norm,
                        color = col,
                        opacity = realOpacity,
                        scale = scale
                    });
                }
            }

            return result;
        }

        public static List<SplatData> FilterIsolatedOutliers(List<SplatData> splats, float radius, int minNeighbors)
        {
            if (splats == null || splats.Count == 0) return splats;
            if (radius <= 0.001f) radius = 0.08f;

            Dictionary<Vector3Int, int> gridCounts = new Dictionary<Vector3Int, int>();

            for (int i = 0; i < splats.Count; i++)
            {
                Vector3 p = splats[i].position;
                Vector3Int cell = new Vector3Int(
                    Mathf.FloorToInt(p.x / radius),
                    Mathf.FloorToInt(p.y / radius),
                    Mathf.FloorToInt(p.z / radius)
                );

                gridCounts.TryGetValue(cell, out int c);
                gridCounts[cell] = c + 1;
            }

            List<SplatData> filtered = new List<SplatData>(splats.Count);

            for (int i = 0; i < splats.Count; i++)
            {
                Vector3 p = splats[i].position;
                Vector3Int cell = new Vector3Int(
                    Mathf.FloorToInt(p.x / radius),
                    Mathf.FloorToInt(p.y / radius),
                    Mathf.FloorToInt(p.z / radius)
                );

                int count = 0;
                for (int dx = -1; dx <= 1; dx++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dz = -1; dz <= 1; dz++)
                        {
                            Vector3Int neighborCell = new Vector3Int(cell.x + dx, cell.y + dy, cell.z + dz);
                            if (gridCounts.TryGetValue(neighborCell, out int nVal))
                            {
                                count += nVal;
                            }
                        }
                    }
                }

                if (count >= minNeighbors)
                {
                    filtered.Add(splats[i]);
                }
            }

            return filtered;
        }

        private class PlyProperty
        {
            public string type;
            public string name;
        }

        public static string ReadAsciiLine(BinaryReader reader)
        {
            StringBuilder sb = new StringBuilder();
            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                char c = (char)reader.ReadByte();
                if (c == '\n') break;
                if (c != '\r') sb.Append(c);
            }
            return sb.ToString();
        }

        public static float ReadPropertyValue(BinaryReader br, string typeName)
        {
            switch (typeName.ToLower())
            {
                case "float":
                case "float32":
                    return br.ReadSingle();
                case "double":
                case "float64":
                    return (float)br.ReadDouble();
                case "uchar":
                case "uint8":
                    return br.ReadByte();
                case "int":
                case "int32":
                    return br.ReadInt32();
                case "short":
                case "int16":
                    return br.ReadInt16();
                case "ushort":
                case "uint16":
                    return br.ReadUInt16();
                default:
                    return br.ReadSingle();
            }
        }

    }
}
