// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Gsplat
{
    /// <summary>
    /// Owns a frozen inverse rank and an immutable, storage-ordered GPU selection.
    /// Keep-count changes require no source upload, allocation or GPU readback.
    /// </summary>
    public sealed class Stage1GpuSelection : IDisposable
    {
        const int GroupSize = 256;
        readonly ComputeShader m_shader;
        readonly int m_sourceCount;
        readonly int m_countKernel, m_scanKernel, m_addKernel, m_scatterKernel, m_seedKernel;
        readonly GraphicsBuffer m_inverseRank;
        readonly List<GraphicsBuffer> m_counts = new();
        readonly List<GraphicsBuffer> m_offsets = new();
        bool m_dirty;

        static readonly int SourceCountId = Shader.PropertyToID("_SourceCount");
        static readonly int KeepCountId = Shader.PropertyToID("_KeepCount");
        static readonly int InverseRankId = Shader.PropertyToID("_InverseRank");
        static readonly int BlockCountsId = Shader.PropertyToID("_BlockCounts");
        static readonly int BlockOffsetsId = Shader.PropertyToID("_BlockOffsets");
        static readonly int SelectedIdsId = Shader.PropertyToID("_SelectedIds");
        static readonly int ScanCountId = Shader.PropertyToID("_ScanCount");
        static readonly int ScanInputId = Shader.PropertyToID("_ScanInput");
        static readonly int ScanOffsetsId = Shader.PropertyToID("_ScanOffsets");
        static readonly int ScanSumsId = Shader.PropertyToID("_ScanSums");
        static readonly int ParentOffsetsId = Shader.PropertyToID("_ParentOffsets");
        static readonly int SortOrderId = Shader.PropertyToID("_SortOrder");

        public int SelectedCount { get; private set; }
        public int EligibleCount { get; private set; }
        public GraphicsBuffer SelectedIds { get; private set; }

        public Stage1GpuSelection(uint sourceCount, ComputeShader shader)
        {
            if (sourceCount > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(sourceCount));
            if (!shader) throw new ArgumentNullException(nameof(shader));
            m_sourceCount = (int)sourceCount;
            m_shader = shader;
            m_countKernel = shader.FindKernel("CountSelected");
            m_scanKernel = shader.FindKernel("ScanBlocks");
            m_addKernel = shader.FindKernel("AddParentOffsets");
            m_scatterKernel = shader.FindKernel("ScatterSelected");
            m_seedKernel = shader.FindKernel("SeedSortOrder");
            foreach (int kernel in new[] { m_countKernel, m_scanKernel, m_addKernel, m_scatterKernel, m_seedKernel })
                if (!shader.IsSupported(kernel))
                    throw new NotSupportedException("Stage 1 GPU selection is not supported by the active graphics device.");

            m_inverseRank = Buffer(Math.Max(1, m_sourceCount), "Stage1InverseRank");
            SelectedIds = Buffer(Math.Max(1, m_sourceCount), "Stage1SelectedStorageIds");
            int length = Math.Max(1, Groups(m_sourceCount));
            while (true)
            {
                m_counts.Add(Buffer(length, "Stage1BlockCounts"));
                m_offsets.Add(Buffer(length, "Stage1BlockOffsets"));
                length = Groups(length);
                if (length == 1)
                {
                    // The final scan writes its sum here; no readback is needed because k is known.
                    m_counts.Add(Buffer(1, "Stage1TotalSelected"));
                    break;
                }
            }
        }

        static GraphicsBuffer Buffer(int count, string name) =>
            new(GraphicsBuffer.Target.Structured, count, sizeof(uint)) { name = name };

        static int Groups(int count) => (int)(((long)count + GroupSize - 1) / GroupSize);

        public void SetRank(uint[] rankToStorage)
        {
            if (rankToStorage == null) throw new ArgumentNullException(nameof(rankToStorage));
            if (rankToStorage.Length > m_sourceCount)
                throw new ArgumentException("The eligible rank cannot exceed the source count.", nameof(rankToStorage));

            // Validate completely before publishing a new rank. The inverse also makes the
            // selection independent of later mutations of the caller's rank array.
            var inverse = new uint[Math.Max(1, m_sourceCount)];
            Array.Fill(inverse, uint.MaxValue);
            for (int rank = 0; rank < rankToStorage.Length; rank++)
            {
                uint storage = rankToStorage[rank];
                if (storage >= m_sourceCount || inverse[storage] != uint.MaxValue)
                    throw new ArgumentException("Rank entries must be unique valid storage IDs.", nameof(rankToStorage));
                inverse[storage] = (uint)rank;
            }
            m_inverseRank.SetData(inverse);
            EligibleCount = rankToStorage.Length;
            SelectedCount = EligibleCount;
            m_dirty = true;
        }

        public void SetKeepCount(int count)
        {
            if (count < 0 || count > EligibleCount) throw new ArgumentOutOfRangeException(nameof(count));
            if (SelectedCount == count) return;
            SelectedCount = count;
            m_dirty = true;
        }

        /// <summary>Record a count/prefix/scatter rebuild only when the selected prefix changed.</summary>
        public void RecordSelection(CommandBuffer cmd)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (!m_dirty) return;
            m_dirty = false;
            if (SelectedCount == 0) return;

            cmd.BeginSample("Stage1.CompactSelection");
            cmd.SetComputeIntParam(m_shader, SourceCountId, m_sourceCount);
            cmd.SetComputeIntParam(m_shader, KeepCountId, SelectedCount);
            cmd.SetComputeBufferParam(m_shader, m_countKernel, InverseRankId, m_inverseRank);
            cmd.SetComputeBufferParam(m_shader, m_countKernel, BlockCountsId, m_counts[0]);
            cmd.DispatchCompute(m_shader, m_countKernel, Groups(m_sourceCount), 1, 1);

            for (int level = 0; level < m_offsets.Count; level++)
            {
                cmd.SetComputeIntParam(m_shader, ScanCountId, m_counts[level].count);
                cmd.SetComputeBufferParam(m_shader, m_scanKernel, ScanInputId, m_counts[level]);
                cmd.SetComputeBufferParam(m_shader, m_scanKernel, ScanOffsetsId, m_offsets[level]);
                cmd.SetComputeBufferParam(m_shader, m_scanKernel, ScanSumsId, m_counts[level + 1]);
                cmd.DispatchCompute(m_shader, m_scanKernel, Groups(m_counts[level].count), 1, 1);
            }
            for (int level = m_offsets.Count - 2; level >= 0; level--)
            {
                cmd.SetComputeIntParam(m_shader, ScanCountId, m_offsets[level].count);
                cmd.SetComputeBufferParam(m_shader, m_addKernel, ScanOffsetsId, m_offsets[level]);
                cmd.SetComputeBufferParam(m_shader, m_addKernel, ParentOffsetsId, m_offsets[level + 1]);
                cmd.DispatchCompute(m_shader, m_addKernel, Groups(m_offsets[level].count), 1, 1);
            }

            cmd.SetComputeBufferParam(m_shader, m_scatterKernel, InverseRankId, m_inverseRank);
            cmd.SetComputeBufferParam(m_shader, m_scatterKernel, BlockOffsetsId, m_offsets[0]);
            cmd.SetComputeBufferParam(m_shader, m_scatterKernel, SelectedIdsId, SelectedIds);
            cmd.DispatchCompute(m_shader, m_scatterKernel, Groups(m_sourceCount), 1, 1);
            cmd.EndSample("Stage1.CompactSelection");
        }

        /// <summary>Reset the mutable sort payload for every camera so depth ties have canonical order.</summary>
        public void SeedOrder(CommandBuffer cmd, GraphicsBuffer order)
        {
            if (order == null || order.count < SelectedCount)
                throw new ArgumentException("Sort buffer is too small for the selected prefix.", nameof(order));
            RecordSelection(cmd);
            if (SelectedCount == 0) return;
            cmd.SetComputeIntParam(m_shader, KeepCountId, SelectedCount);
            cmd.SetComputeBufferParam(m_shader, m_seedKernel, SelectedIdsId, SelectedIds);
            cmd.SetComputeBufferParam(m_shader, m_seedKernel, SortOrderId, order);
            cmd.DispatchCompute(m_shader, m_seedKernel, Groups(SelectedCount), 1, 1);
        }

        public void Dispose()
        {
            m_inverseRank?.Dispose();
            SelectedIds?.Dispose();
            SelectedIds = null;
            foreach (var buffer in m_counts) buffer.Dispose();
            foreach (var buffer in m_offsets) buffer.Dispose();
            m_counts.Clear();
            m_offsets.Clear();
        }
    }
}
