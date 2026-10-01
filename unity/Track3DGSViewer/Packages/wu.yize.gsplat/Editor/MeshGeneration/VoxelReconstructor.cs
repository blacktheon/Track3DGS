using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEditor;

namespace Gsplat.Editor
{
    public static class VoxelReconstructor
    {

        public static MeshData ComputeMesh(List<SplatData> splats, int resolution, float threshold)
        {
            return ReconstructVoxelMarchingCubes(splats, resolution, threshold);
        }

        public static MeshData ReconstructVoxelMarchingCubes(List<SplatData> splats, int resolution, float threshold)
        {
            // Calculate Point Bounding Box
            Bounds bounds = new Bounds(splats[0].position, Vector3.zero);
            foreach (var s in splats)
            {
                bounds.Encapsulate(s.position);
            }
            bounds.Expand(0.5f);

            Vector3 min = bounds.min;
            Vector3 size = bounds.size;
            Vector3 voxelStep = new Vector3(size.x / (resolution - 1), size.y / (resolution - 1), size.z / (resolution - 1));

            const int CHUNK_SIZE = 48; // 48x48x48 voxel block per streaming chunk
            int numChunksX = Mathf.CeilToInt((float)(resolution - 1) / CHUNK_SIZE);
            int numChunksY = Mathf.CeilToInt((float)(resolution - 1) / CHUNK_SIZE);
            int numChunksZ = Mathf.CeilToInt((float)(resolution - 1) / CHUNK_SIZE);

            // Spatial binning of splats into chunks for fast density evaluation
            List<SplatData>[,,] chunkSplats = new List<SplatData>[numChunksX, numChunksY, numChunksZ];
            for (int cx = 0; cx < numChunksX; cx++)
                for (int cy = 0; cy < numChunksY; cy++)
                    for (int cz = 0; cz < numChunksZ; cz++)
                        chunkSplats[cx, cy, cz] = new List<SplatData>();

            float chunkDimX = CHUNK_SIZE * voxelStep.x;
            float chunkDimY = CHUNK_SIZE * voxelStep.y;
            float chunkDimZ = CHUNK_SIZE * voxelStep.z;

            foreach (var s in splats)
            {
                Vector3 relPos = s.position - min;
                int cx = Mathf.Clamp(Mathf.FloorToInt(relPos.x / chunkDimX), 0, numChunksX - 1);
                int cy = Mathf.Clamp(Mathf.FloorToInt(relPos.y / chunkDimY), 0, numChunksY - 1);
                int cz = Mathf.Clamp(Mathf.FloorToInt(relPos.z / chunkDimZ), 0, numChunksZ - 1);

                chunkSplats[cx, cy, cz].Add(s);
            }

            MeshData data = new MeshData();

            // Reusable single chunk transmittance buffer (takes only ~1.1 MB RAM!)
            float[,,] chunkTransmittance = new float[CHUNK_SIZE + 1, CHUNK_SIZE + 1, CHUNK_SIZE + 1];

            // Precompute exponential falloff weights for dx, dy, dz from -1 to 1 (max distSq = 3)
            float[] weightLookup = new float[4];
            for (int i = 0; i < 4; i++)
            {
                weightLookup[i] = Mathf.Exp(-i * 0.5f);
            }

            for (int cx = 0; cx < numChunksX; cx++)
            {
                for (int cy = 0; cy < numChunksY; cy++)
                {
                    for (int cz = 0; cz < numChunksZ; cz++)
                    {
                        // Gather splats from current chunk + 26 adjacent neighbor chunks
                        List<SplatData> localSplats = new List<SplatData>();
                        for (int ncx = Mathf.Max(0, cx - 1); ncx <= Mathf.Min(numChunksX - 1, cx + 1); ncx++)
                            for (int ncy = Mathf.Max(0, cy - 1); ncy <= Mathf.Min(numChunksY - 1, cy + 1); ncy++)
                                for (int ncz = Mathf.Max(0, cz - 1); ncz <= Mathf.Min(numChunksZ - 1, cz + 1); ncz++)
                                    localSplats.AddRange(chunkSplats[ncx, ncy, ncz]);

                        if (localSplats.Count == 0) continue;

                        // Initialize chunk transmittance to 1.0 (completely transparent)
                        for (int tx = 0; tx <= CHUNK_SIZE; tx++)
                            for (int ty = 0; ty <= CHUNK_SIZE; ty++)
                                for (int tz = 0; tz <= CHUNK_SIZE; tz++)
                                    chunkTransmittance[tx, ty, tz] = 1.0f;

                        int startX = cx * CHUNK_SIZE;
                        int startY = cy * CHUNK_SIZE;
                        int startZ = cz * CHUNK_SIZE;

                        int limitX = Mathf.Min(startX + CHUNK_SIZE, resolution - 1);
                        int limitY = Mathf.Min(startY + CHUNK_SIZE, resolution - 1);
                        int limitZ = Mathf.Min(startZ + CHUNK_SIZE, resolution - 1);

                        int countX = limitX - startX + 1;
                        int countY = limitY - startY + 1;
                        int countZ = limitZ - startZ + 1;

                        Vector3 chunkMin = min + new Vector3(startX * voxelStep.x, startY * voxelStep.y, startZ * voxelStep.z);

                        // Accumulate optical transmittance into local chunk buffer
                        foreach (var s in localSplats)
                        {
                            Vector3 relPos = s.position - chunkMin;
                            int gx = Mathf.FloorToInt(relPos.x / voxelStep.x);
                            int gy = Mathf.FloorToInt(relPos.y / voxelStep.y);
                            int gz = Mathf.FloorToInt(relPos.z / voxelStep.z);

                            for (int dx = -1; dx <= 1; dx++)
                            {
                                for (int dy = -1; dy <= 1; dy++)
                                {
                                    for (int dz = -1; dz <= 1; dz++)
                                    {
                                        int nx = gx + dx;
                                        int ny = gy + dy;
                                        int nz = gz + dz;

                                        if (nx >= 0 && nx < countX && ny >= 0 && ny < countY && nz >= 0 && nz < countZ)
                                        {
                                            int distSq = (dx * dx + dy * dy + dz * dz);
                                            float weight = Mathf.Clamp01(s.opacity * weightLookup[distSq]);
                                            chunkTransmittance[nx, ny, nz] *= (1.0f - weight);
                                        }
                                    }
                                }
                            }
                        }

                        // March Cubes over local chunk
                        for (int x = 0; x < countX - 1; x++)
                        {
                            for (int y = 0; y < countY - 1; y++)
                            {
                                for (int z = 0; z < countZ - 1; z++)
                                {
                                    Vector3[] cubeCorners = new Vector3[8];
                                    float[] cubeVals = new float[8];

                                    for (int i = 0; i < 8; i++)
                                    {
                                        int lx = x + MarchingCubesTables.CornerOffsets[i, 0];
                                        int ly = y + MarchingCubesTables.CornerOffsets[i, 1];
                                        int lz = z + MarchingCubesTables.CornerOffsets[i, 2];

                                        int worldX = startX + lx;
                                        int worldY = startY + ly;
                                        int worldZ = startZ + lz;

                                        cubeCorners[i] = min + new Vector3(worldX * voxelStep.x, worldY * voxelStep.y, worldZ * voxelStep.z);
                                        // Combined opacity = 1.0 - transmittance
                                        cubeVals[i] = 1.0f - chunkTransmittance[lx, ly, lz];
                                    }

                                    int cubeIndex = 0;
                                    for (int i = 0; i < 8; i++)
                                    {
                                        if (cubeVals[i] >= threshold) cubeIndex |= (1 << i);
                                    }

                                    int edgeFlags = MarchingCubesTables.EdgeTable[cubeIndex];
                                    if (edgeFlags == 0) continue;

                                    Vector3[] edgeVerts = new Vector3[12];

                                    for (int i = 0; i < 12; i++)
                                    {
                                        if ((edgeFlags & (1 << i)) != 0)
                                        {
                                            int a = MarchingCubesTables.EdgeConnection[i, 0];
                                            int b = MarchingCubesTables.EdgeConnection[i, 1];

                                            float mu = (threshold - cubeVals[a]) / (cubeVals[b] - cubeVals[a] + 1e-6f);
                                            mu = Mathf.Clamp01(mu);

                                            edgeVerts[i] = Vector3.Lerp(cubeCorners[a], cubeCorners[b], mu);
                                        }
                                    }

                                    for (int i = 0; i + 2 < 16 && MarchingCubesTables.TriTable[cubeIndex, i] != -1; i += 3)
                                    {
                                        int e0 = MarchingCubesTables.TriTable[cubeIndex, i];
                                        int e1 = MarchingCubesTables.TriTable[cubeIndex, i + 1];
                                        int e2 = MarchingCubesTables.TriTable[cubeIndex, i + 2];

                                        if (e0 < 0 || e1 < 0 || e2 < 0) break;

                                        int idx1 = data.vertices.Count;
                                        data.vertices.Add(edgeVerts[e0]);
                                        data.vertices.Add(edgeVerts[e1]);
                                        data.vertices.Add(edgeVerts[e2]);

                                        data.triangles.Add(idx1);
                                        data.triangles.Add(idx1 + 1);
                                        data.triangles.Add(idx1 + 2);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return data;
        }



    }
}
