// Copyright (c) 2025 Yize Wu
// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Vector3 = UnityEngine.Vector3;

namespace Gsplat
{
    public class GsplatRendererImpl
    {
        public uint SplatCount { get; private set; }

        MaterialPropertyBlock m_propertyBlock;
        GsplatAsset m_gsplatAsset;
#if UNITY_EDITOR
        readonly Dictionary<Material, Material> m_editorMaterials = new();
#endif
        public uint m_remainingCount = 0;
        public Bounds m_bounds;
        ulong m_gsplatAssetID;

        public GsplatResource GsplatResource;
        public GraphicsBuffer OrderBuffer { get; private set; }
        public GraphicsBuffer CutoutsBuffer { get; private set; }
        public GraphicsBuffer OrderSizeBuffer { get; private set; }
        public GraphicsBuffer BoundsBuffer { get; private set; }
        public ISorterResource SorterResource { get; private set; }

        static readonly int k_orderBuffer = Shader.PropertyToID("_OrderBuffer");
        static readonly int k_matrixM = Shader.PropertyToID("_MATRIX_M");
        static readonly int k_splatInstanceSize = Shader.PropertyToID("_SplatInstanceSize");
        static readonly int k_splatCount = Shader.PropertyToID("_SplatCount");
        static readonly int k_gammaToLinear = Shader.PropertyToID("_GammaToLinear");
        static readonly int k_shDegree = Shader.PropertyToID("_SHDegree");
        static readonly int k_brightness = Shader.PropertyToID("_Brightness");
        static readonly int k_scaleFactor = Shader.PropertyToID("_ScaleFactor");

        uint m_framesBeforeRecomputeSort = 0;
        uint m_sortsBeforeRecomputeCutouts = 0;
        public bool ComputeSortRequired = true;
        public bool ComputeCutoutsRequired = true;
        Dictionary<ulong, (Vector3, Vector3)> m_prevCamTransforms;

        GsplatCutout.ShaderData[] m_cutoutsData;
        uint m_prevSplatCount;
        Stage1GpuSelection m_stage1Selection;
        Stage1GpuSelection m_stage1PendingSelection;
        bool m_stage1ClearPending;
        int m_stage1KeepCount;

        public bool HasStage1Selection => m_stage1Selection != null || m_stage1PendingSelection != null;
        public int Stage1SelectedCount => HasStage1Selection ? m_stage1KeepCount : (int)m_remainingCount;
        public int Stage1EligibleCount => m_stage1PendingSelection?.EligibleCount ?? m_stage1Selection?.EligibleCount ?? (int)SplatCount;
        public bool HasStage1Walls { get; private set; }

        const int Stage1MaxWalls = 16;
        static readonly int k_stage1WallCount = Shader.PropertyToID("_Stage1WallCount");
        static readonly int k_stage1WorldToLocal = Shader.PropertyToID("_Stage1WorldToLocal");
        static readonly int k_stage1Bounds = Shader.PropertyToID("_Stage1BoundsMinMaxXZ");
        static readonly int k_stage1Planes = Shader.PropertyToID("_Stage1WorldPlanes");
        static readonly int k_stage1RearDepths = Shader.PropertyToID("_Stage1RearDepths");

        public GsplatRendererImpl(uint splatCount)
        {
            SplatCount = splatCount;
            m_prevCamTransforms = new Dictionary<ulong, (Vector3, Vector3)>();
            CreateResources(splatCount);
            CreatePropertyBlock();
        }

        public void RecreateResources(uint splatCount)
        {
            if (SplatCount == splatCount)
                return;
            Dispose();
            SplatCount = splatCount;
            CreateResources(splatCount);
            CreatePropertyBlock();
        }

        public void ComputeDepth(CommandBuffer cmd, Matrix4x4 matrixMv)
        {
            if (m_remainingCount == 0) return;
            if (m_stage1Selection != null)
            {
                // The selected list remains unchanged by sorting; every camera starts with
                // the same ascending storage IDs, including cameras with equal depths.
                m_stage1Selection.SeedOrder(cmd, OrderBuffer);
                SorterResource.Initialized = true;
            }
            m_gsplatAsset.ComputeDepth(cmd, matrixMv, SorterResource, GsplatResource, m_remainingCount);
        }

        public void SetStage1Rank(uint[] rankToStorage)
        {
            if (GsplatResource == null || GsplatResource.UploadedCount != SplatCount)
                throw new InvalidOperationException("Stage 1 selection requires the complete source upload.");
            if (m_cutoutsData.Length != 0)
                throw new NotSupportedException("Stage 1 exact selection cannot be combined with Gsplat cutouts.");

            // A replacement rank must not mutate the buffers referenced by this frame's
            // already submitted draw. Activate it at the next draw boundary instead.
            var selection = new Stage1GpuSelection(SplatCount, Resources.Load<ComputeShader>("Stage1Selection"));
            try { selection.SetRank(rankToStorage); }
            catch { selection.Dispose(); throw; }
            m_stage1PendingSelection?.Dispose();
            m_stage1PendingSelection = selection;
            m_stage1ClearPending = false;
            m_stage1KeepCount = selection.EligibleCount;
            ForceRefresh();
            GsplatSorter.Instance.InvalidateStage1GlobalRender();
        }

        public void SetStage1KeepCount(int count)
        {
            if (!HasStage1Selection || m_stage1ClearPending)
                throw new InvalidOperationException("Load a Stage 1 rank before changing its keep count.");
            if (count < 0 || count > Stage1EligibleCount) throw new ArgumentOutOfRangeException(nameof(count));
            if (m_stage1KeepCount == count) return;
            // Commit before the next draw submission, together with m_remainingCount.
            // An input event after Update must not change the sort count of an already submitted draw.
            m_stage1KeepCount = count;
            ForceRefresh();
        }

        public void ClearStage1Selection()
        {
            m_stage1PendingSelection?.Dispose();
            m_stage1PendingSelection = null;
            m_stage1ClearPending = true;
            ForceRefresh();
        }

        void ReleaseStage1Selection()
        {
            m_stage1Selection?.Dispose();
            m_stage1Selection = null;
            m_stage1PendingSelection?.Dispose();
            m_stage1PendingSelection = null;
            m_stage1ClearPending = false;
            m_stage1KeepCount = 0;
            if (SorterResource != null) SorterResource.Initialized = false;
            ForceRefresh();
        }

        public void SetStage1Walls(Stage1Wall[] walls)
        {
            if (walls == null) throw new ArgumentNullException(nameof(walls));
            if (walls.Length > Stage1MaxWalls) throw new ArgumentException("At most 16 Stage 1 walls are supported.", nameof(walls));
            foreach (var wall in walls) wall.Validate();
            var transforms = new Matrix4x4[Stage1MaxWalls];
            var bounds = new Vector4[Stage1MaxWalls];
            var planes = new Vector4[Stage1MaxWalls];
            var depths = new float[Stage1MaxWalls];
            for (int i = 0; i < walls.Length; i++)
            {
                transforms[i] = walls[i].WorldToLocal;
                bounds[i] = walls[i].BoundsMinMaxXZ;
                planes[i] = walls[i].WorldPlane;
                depths[i] = walls[i].RearDepth;
            }
            m_propertyBlock.SetMatrixArray(k_stage1WorldToLocal, transforms);
            m_propertyBlock.SetVectorArray(k_stage1Bounds, bounds);
            m_propertyBlock.SetVectorArray(k_stage1Planes, planes);
            m_propertyBlock.SetFloatArray(k_stage1RearDepths, depths);
            m_propertyBlock.SetInteger(k_stage1WallCount, walls.Length);
            HasStage1Walls = walls.Length != 0;
            if (HasStage1Walls) GsplatSorter.Instance.InvalidateStage1GlobalRender();
        }

        public void ClearStage1Walls()
        {
            m_propertyBlock.SetInteger(k_stage1WallCount, 0);
            HasStage1Walls = false;
        }

        Bounds ExtractBounds()
        {
            uint[] boundsData = new uint[6];
            BoundsBuffer.GetData(boundsData);

            Bounds bounds = default;
            Vector3 bmin = new(GsplatUtils.SortableUintToFloat(boundsData[0]),
                GsplatUtils.SortableUintToFloat(boundsData[1]), GsplatUtils.SortableUintToFloat(boundsData[2]));
            Vector3 bmax = new(GsplatUtils.SortableUintToFloat(boundsData[3]),
                GsplatUtils.SortableUintToFloat(boundsData[4]), GsplatUtils.SortableUintToFloat(boundsData[5]));
            bounds.SetMinMax(bmin, bmax);

            if (bounds.extents.sqrMagnitude < 0.01)
                bounds.extents = new Vector3(0.1f, 0.1f, 0.1f);
            return bounds;
        }

        uint ExtractOrderSize(GraphicsBuffer orderBuffer)
        {
            GraphicsBuffer.CopyCount(orderBuffer, OrderSizeBuffer, 0);
            uint[] count = new uint[1];
            OrderSizeBuffer.GetData(count);
            return count[0];
        }

        public void DispatchInitOrder(GsplatCutout[] cutouts, Matrix4x4 matrixWorld, bool cutoutsUpdateBounds)
        {
            if (m_stage1ClearPending) ReleaseStage1Selection();
            if (m_stage1PendingSelection != null)
            {
                m_stage1Selection?.Dispose();
                m_stage1Selection = m_stage1PendingSelection;
                m_stage1PendingSelection = null;
            }
            if (m_stage1Selection != null)
            {
                if (cutouts.Length != 0)
                    throw new NotSupportedException("Stage 1 exact selection cannot be combined with Gsplat cutouts.");
                if (GsplatResource.UploadedCount != SplatCount)
                    throw new InvalidOperationException("Stage 1 source upload became incomplete.");
                m_stage1Selection.SetKeepCount(m_stage1KeepCount);
                m_remainingCount = (uint)m_stage1KeepCount;
                m_bounds = m_gsplatAsset.Bounds;
                SorterResource.Initialized = true;
                return;
            }
            if (cutouts.Length == 0)
            {
                if (m_cutoutsData.Length > 0)
                    SorterResource.Initialized = false;
                m_cutoutsData = Array.Empty<GsplatCutout.ShaderData>();
                m_remainingCount = GsplatResource.UploadedCount;
                m_bounds = m_gsplatAsset.Bounds;
                return;
            }

            if (!ComputeCutoutsRequired)
                return;

            SorterResource.Initialized = true;

            var cutoutsUnchanged = m_cutoutsData.Length == cutouts.Length;
            var updatedCutoutsData = new GsplatCutout.ShaderData[cutouts.Length];
            for (int i = 0; i != cutouts.Length; i++)
            {
                updatedCutoutsData[i] = cutouts[i].GetShaderData(matrixWorld);
                if (cutoutsUnchanged)
                    if (updatedCutoutsData[i].matrix != m_cutoutsData[i].matrix ||
                        updatedCutoutsData[i].typeAndFlags != m_cutoutsData[i].typeAndFlags)
                        cutoutsUnchanged = false;
            }

            if (cutoutsUnchanged && m_prevSplatCount == GsplatResource.UploadedCount)
                return;

            m_prevSplatCount = GsplatResource.UploadedCount;
            m_cutoutsData = updatedCutoutsData;
            CutoutsBuffer = m_gsplatAsset.UpdateCutoutsBuffer(CutoutsBuffer, m_cutoutsData);
            if (cutoutsUpdateBounds)
                m_gsplatAsset.UpdateBoundsBuffer(BoundsBuffer);
            m_gsplatAsset.InitOrder(SorterResource, GsplatResource, cutoutsUpdateBounds);
            m_remainingCount = ExtractOrderSize(SorterResource.OrderBuffer);
            m_bounds = cutoutsUpdateBounds ? ExtractBounds() : m_gsplatAsset.Bounds;
        }

        public void BindGsplatAsset(GsplatAsset gsplatAsset, bool asyncUpload = false)
        {
            Debug.Assert(m_gsplatAssetID == 0);
            m_gsplatAssetID = GsplatUtils.GetObjectId(gsplatAsset);
            m_gsplatAsset = gsplatAsset;
            GsplatResource = GsplatResourceManager.Get(gsplatAsset);
            gsplatAsset.SetupMaterialPropertyBlock(m_propertyBlock, GsplatResource);
            if (asyncUpload)
                gsplatAsset.UploadDataAsync(GsplatResource);
            else
                gsplatAsset.UploadData(GsplatResource);
        }

        public void ReleaseGsplatAsset()
        {
            // Resource teardown cannot leave pending buffers alive for a later Update.
            ReleaseStage1Selection();
#if UNITY_EDITOR
            foreach (var material in m_editorMaterials.Values)
                if (material) UnityEngine.Object.DestroyImmediate(material);
            m_editorMaterials.Clear();
#endif
            GsplatResourceManager.Release(m_gsplatAssetID);
            GsplatResource = null;
            m_gsplatAsset = null;
            m_gsplatAssetID = 0;
        }

        void CreateResources(uint splatCount)
        {
            OrderBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Append, (int)splatCount, sizeof(uint));
            SorterResource = GsplatSorter.Instance.CreateSorterResource(splatCount, OrderBuffer);
            m_cutoutsData = Array.Empty<GsplatCutout.ShaderData>();
            CutoutsBuffer = null;
            OrderSizeBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, sizeof(uint));
            BoundsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 6, sizeof(uint));
        }

        void CreatePropertyBlock()
        {
            m_propertyBlock ??= new MaterialPropertyBlock();
            m_propertyBlock.SetBuffer(k_orderBuffer, OrderBuffer);
            ClearStage1Walls();
        }

        public void Dispose()
        {
            ReleaseGsplatAsset();
            OrderBuffer?.Dispose();
            OrderBuffer = null;
            SorterResource?.Dispose();
            SorterResource = null;
            CutoutsBuffer?.Dispose();
            CutoutsBuffer = null;
            OrderSizeBuffer?.Dispose();
            OrderSizeBuffer = null;
            BoundsBuffer?.Dispose();
            BoundsBuffer = null;
        }

        public void ForceRefresh()
        {
            m_framesBeforeRecomputeSort = 0;
            m_sortsBeforeRecomputeCutouts = 0;
        }

        public void RefreshOnCameraMove()
        {
            foreach (var cam in Camera.allCameras)
            {
                var id = GsplatUtils.GetObjectId(cam);
                if (m_prevCamTransforms.TryGetValue(id, out (Vector3, Vector3) prevCamTransform))
                {
                    (Vector3 prevCamPos, Vector3 prevCamRot) = prevCamTransform;

                    if ((cam.transform.position - prevCamPos).magnitude >
                        GsplatSettings.Instance.CameraTranslationRefreshTreshold
                        || (cam.transform.eulerAngles - prevCamRot).magnitude >
                        GsplatSettings.Instance.CameraRotationRefreshTreshold)
                    {
                        m_prevCamTransforms[id] = (cam.transform.position, cam.transform.eulerAngles);
                        ForceRefresh();
                    }
                }
                else
                {
                    m_prevCamTransforms.Add(id, (cam.transform.position, cam.transform.eulerAngles));
                    ForceRefresh();
                }
            }
        }

        public void EvaluateRefreshRequired(GsplatRenderer.GsplatSortMode mode, uint sortRefreshRate,
            uint cutoutsRefreshRate)
        {
            if (mode == GsplatRenderer.GsplatSortMode.Always)
            {
                sortRefreshRate = 0;
                cutoutsRefreshRate = 0;
            }

            if (mode == GsplatRenderer.GsplatSortMode.SortEveryNFrames)
            {
                cutoutsRefreshRate = 0;
            }

            RefreshOnCameraMove();

            ComputeSortRequired = false;
            ComputeCutoutsRequired = false;

            if (m_framesBeforeRecomputeSort == 0)
            {
                m_framesBeforeRecomputeSort = sortRefreshRate;
                ComputeSortRequired = true;
                if (m_sortsBeforeRecomputeCutouts == 0)
                {
                    m_sortsBeforeRecomputeCutouts = cutoutsRefreshRate;
                    ComputeCutoutsRequired = true;
                }
                else
                    m_sortsBeforeRecomputeCutouts -= 1;
            }
            else
                m_framesBeforeRecomputeSort -= 1;

            // Each camera gets an independently seeded order. Sorting must follow that seed,
            // even when the renderer's ordinary frame-skipping mode is enabled.
            if (HasStage1Selection) ComputeSortRequired = true;
        }

        /// <summary>
        /// Render the splats.
        /// </summary>
        /// <param name="transform">Object transform.</param>
        /// <param name="layer">Layer used for rendering.</param>
        /// <param name="gammaToLinear">Covert color space from Gamma to Linear.</param>
        /// <param name="shDegree">Order of SH coefficients used for rendering. The final value is capped by the SHBands property.</param>
        /// <param name="brightness">Brightness color scaling.</param>
        /// <param name="scaleFactor">Splats uv scaling factor, reduce splat size while trying to keep visual fidelity.</param>
        /// <param name="renderOrder">Manual render order placement of the gsplat. The final value is capped by the maximum render order setting.</param>
        public void Render(Transform transform, int layer, bool gammaToLinear = false, int shDegree = 3,
            float brightness = 1.0f, float scaleFactor = 1.0f, uint renderOrder = 0)
        {
            if (m_remainingCount <= 0)
                return;

            SetRenderProperties(transform, gammaToLinear, shDegree, brightness, scaleFactor);

            uint order = Math.Clamp(renderOrder, 0, GsplatSettings.Instance.MaxRenderOrder - 1);
            var rp = new RenderParams(m_gsplatAsset.Materials[order])
            {
                worldBounds = GsplatUtils.CalcWorldBounds(m_bounds, transform),
                matProps = m_propertyBlock,
                layer = layer
            };

            Graphics.RenderMeshPrimitives(rp, GsplatSettings.Instance.Mesh, 0,
                Mathf.CeilToInt(m_remainingCount / (float)GsplatSettings.Instance.SplatInstanceSize));
        }

        void SetRenderProperties(Transform transform, bool gammaToLinear, int shDegree, float brightness, float scaleFactor)
        {
            m_propertyBlock.SetInteger(k_splatCount, (int)m_remainingCount);
            m_propertyBlock.SetInteger(k_gammaToLinear, gammaToLinear ? 1 : 0);
            m_propertyBlock.SetInteger(k_splatInstanceSize, (int)GsplatSettings.Instance.SplatInstanceSize);
            m_propertyBlock.SetInteger(k_shDegree, Math.Min(m_gsplatAsset.SHBands, shDegree));
            m_propertyBlock.SetFloat(k_brightness, brightness);
            m_propertyBlock.SetFloat(k_scaleFactor, scaleFactor);
            m_propertyBlock.SetMatrix(k_matrixM, transform.localToWorldMatrix);
        }

#if UNITY_EDITOR
        /// <summary>Record one camera-local draw after that camera's depth sort, using its existing targets.</summary>
        public void RenderEditorPreview(CommandBuffer commandBuffer, Transform transform, bool gammaToLinear,
            int shDegree, float brightness, float scaleFactor, uint renderOrder)
        {
            if (commandBuffer == null) throw new ArgumentNullException(nameof(commandBuffer));
            if (m_remainingCount == 0) return;
            SetRenderProperties(transform, gammaToLinear, shDegree, brightness, scaleFactor);
            uint order = Math.Clamp(renderOrder, 0, GsplatSettings.Instance.MaxRenderOrder - 1);
            var sourceMaterial = m_gsplatAsset.Materials[order];
            if (!m_editorMaterials.TryGetValue(sourceMaterial, out var material) || !material)
            {
                material = new Material(sourceMaterial)
                {
                    name = sourceMaterial.name + " (Editor preview)",
                    hideFlags = HideFlags.HideAndDontSave,
                    enableInstancing = true
                };
                m_editorMaterials[sourceMaterial] = material;
            }
            commandBuffer.DrawMeshInstancedProcedural(GsplatSettings.Instance.Mesh, 0, material, 0,
                Mathf.CeilToInt(m_remainingCount / (float)GsplatSettings.Instance.SplatInstanceSize), m_propertyBlock);
        }
#endif
    }
}
