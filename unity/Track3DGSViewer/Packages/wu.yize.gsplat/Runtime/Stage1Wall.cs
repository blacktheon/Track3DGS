// SPDX-License-Identifier: MIT

using System;
using UnityEngine;

namespace Gsplat
{
    /// <summary>Finite front-facing plane with a world-unit rear band, evaluated at splat centers.</summary>
    [Serializable]
    public struct Stage1Wall
    {
        public Matrix4x4 WorldToLocal;
        /// <summary>xmin, xmax, zmin, zmax in the wall's local frame.</summary>
        public Vector4 BoundsMinMaxXZ;
        /// <summary>Normalized world normal xyz and plane offset w; front has positive signed distance.</summary>
        public Vector4 WorldPlane;
        public float RearDepth;

        /// <summary>CPU diagnostic equivalent of the per-camera shader visibility profile.</summary>
        public bool BlocksView(Vector3 eyeWorld, Vector3 centerWorld)
        {
            var normal = new Vector3(WorldPlane.x, WorldPlane.y, WorldPlane.z);
            float eyeDistance = Vector3.Dot(normal, eyeWorld) + WorldPlane.w;
            float centerDistance = Vector3.Dot(normal, centerWorld) + WorldPlane.w;
            if (eyeDistance <= 1e-6f || centerDistance >= 0 || centerDistance < -RearDepth) return false;
            float t = eyeDistance / (eyeDistance - centerDistance);
            var local = WorldToLocal.MultiplyPoint3x4(Vector3.LerpUnclamped(eyeWorld, centerWorld, t));
            return local.x >= BoundsMinMaxXZ.x && local.x <= BoundsMinMaxXZ.y &&
                   local.z >= BoundsMinMaxXZ.z && local.z <= BoundsMinMaxXZ.w;
        }

        internal void Validate()
        {
            for (int i = 0; i < 16; i++)
                if (!Finite(WorldToLocal[i])) throw new ArgumentException("A wall transform must be finite.");
            for (int i = 0; i < 4; i++)
                if (!Finite(WorldPlane[i]) || !Finite(BoundsMinMaxXZ[i]))
                    throw new ArgumentException("Wall plane and bounds must be finite.");
            var normal = new Vector3(WorldPlane.x, WorldPlane.y, WorldPlane.z);
            if (Mathf.Abs(normal.sqrMagnitude - 1f) > 1e-3f)
                throw new ArgumentException("The wall world-plane normal must be normalized.");
            if (!Finite(RearDepth) || RearDepth < 0 || BoundsMinMaxXZ.x > BoundsMinMaxXZ.y ||
                BoundsMinMaxXZ.z > BoundsMinMaxXZ.w)
                throw new ArgumentException("Wall bounds and rear depth must be nonnegative ranges.");
        }

        static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
