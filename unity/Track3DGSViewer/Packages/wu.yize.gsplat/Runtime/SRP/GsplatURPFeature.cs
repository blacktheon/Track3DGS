// Originated from the GaussianSplatHDRPPass in aras-p/UnityGaussianSplatting by Aras Pranckevičius
// https://github.com/aras-p/UnityGaussianSplatting/blob/main/package/Runtime/GaussianSplatHDRPPass.cs
// Copyright (c) 2023 Aras Pranckevičius
// Modified by Yize Wu
// Copyright (c) 2025 Yize Wu
// SPDX-License-Identifier: MIT

#if GSPLAT_ENABLE_URP

using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
#if UNITY_6000_0_OR_NEWER
using UnityEngine.Rendering.RenderGraphModule;
#endif

namespace Gsplat
{
    class GsplatURPFeature : ScriptableRendererFeature
    {
        class GsplatRenderPass : ScriptableRenderPass
        {
#if UNITY_6000_0_OR_NEWER
            class PassData
            {
                public UniversalCameraData CameraData;
#if UNITY_EDITOR
                public bool EditorPreview, BackBuffer;
                public TextureHandle Color, Depth;
#endif
            }

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                using var builder = renderGraph.AddUnsafePass(GsplatSorter.k_passName, out PassData passData);
                passData.CameraData = frameData.Get<UniversalCameraData>();
#if UNITY_EDITOR
                passData.EditorPreview = GsplatRenderer.UsesEditorUrpPreview;
                if (passData.EditorPreview)
                {
                    var resources = frameData.Get<UniversalResourceData>();
                    passData.Color = resources.activeColorTexture;
                    passData.Depth = resources.activeDepthTexture;
                    passData.BackBuffer = resources.isActiveTargetBackBuffer;
                    // Preserve the opaque color and depth already produced by this camera.
                    // The splat shader blends into color and depth-tests with ZWrite disabled.
                    builder.UseTexture(passData.Color, AccessFlags.ReadWrite);
                    builder.UseTexture(passData.Depth, AccessFlags.Read);
                }
#endif
                builder.AllowPassCulling(false);
                builder.SetRenderFunc(static (PassData data, UnsafeGraphContext context) =>
                {
                    var commandBuffer = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                    GsplatSorter.Instance.DispatchSort(commandBuffer, data.CameraData.camera);
#if UNITY_EDITOR
                    if (data.EditorPreview)
                    {
                        CoreUtils.SetRenderTarget(commandBuffer, (RTHandle)data.Color, (RTHandle)data.Depth);
                        if (data.BackBuffer) commandBuffer.SetViewport(data.CameraData.camera.pixelRect);
                        GsplatSorter.Instance.DrawEditorPreviews(commandBuffer, data.CameraData.camera);
                    }
#endif
                });
            }
#else
            public CommandBuffer CommandBuffer;
#if UNITY_EDITOR
            public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
            {
                if (GsplatRenderer.UsesEditorUrpPreview)
                    ConfigureTarget(renderingData.cameraData.renderer.cameraColorTargetHandle,
                        renderingData.cameraData.renderer.cameraDepthTargetHandle);
            }
#endif
            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                GsplatSorter.Instance.DispatchSort(CommandBuffer, renderingData.cameraData.camera);
#if UNITY_EDITOR
                if (GsplatRenderer.UsesEditorUrpPreview)
                    GsplatSorter.Instance.DrawEditorPreviews(CommandBuffer, renderingData.cameraData.camera);
#endif
                context.ExecuteCommandBuffer(CommandBuffer);
            }
#endif
        }

        GsplatRenderPass m_pass;
        bool m_hasGsplats;

        public override void Create()
        {
            m_pass = new GsplatRenderPass { renderPassEvent = RenderPassEvent.BeforeRenderingTransparents };
        }

        public override void OnCameraPreCull(ScriptableRenderer renderer, in CameraData cameraData)
        {
#if UNITY_EDITOR
            GsplatSorter.Instance.PrepareEditorPreviews(cameraData.camera);
#endif
            m_hasGsplats = GsplatSorter.Instance.GatherGsplatsForCamera(cameraData.camera);
#if !UNITY_6000_0_OR_NEWER
            m_pass.CommandBuffer ??= new CommandBuffer { name = "SortGsplats" };
            m_pass.CommandBuffer.Clear();
#endif
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (GsplatSorter.Instance.Valid && GsplatSettings.Instance.Valid && m_hasGsplats)
                renderer.EnqueuePass(m_pass);
        }

        protected override void Dispose(bool disposing)
        {
#if !UNITY_6000_0_OR_NEWER
            m_pass.CommandBuffer?.Dispose();
            m_pass.CommandBuffer = null;
#endif
            m_pass = null;
        }
    }
}

#endif
