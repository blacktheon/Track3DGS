// SPDX-License-Identifier: MIT
#ifndef GSPLAT_STAGE1_WALLS_INCLUDED
#define GSPLAT_STAGE1_WALLS_INCLUDED

#define STAGE1_MAX_WALLS 16
int _Stage1WallCount;
float4x4 _Stage1WorldToLocal[STAGE1_MAX_WALLS];
float4 _Stage1BoundsMinMaxXZ[STAGE1_MAX_WALLS];
float4 _Stage1WorldPlanes[STAGE1_MAX_WALLS];
float _Stage1RearDepths[STAGE1_MAX_WALLS];

bool Stage1BlocksCenter(float3 eyeWorld, float3 centerWorld)
{
    for (int i = 0; i < _Stage1WallCount; i++)
    {
        float4 plane = _Stage1WorldPlanes[i];
        float eyeDistance = dot(plane.xyz, eyeWorld) + plane.w;
        float centerDistance = dot(plane.xyz, centerWorld) + plane.w;
        if (eyeDistance <= 1e-6 || centerDistance >= 0 || centerDistance < -_Stage1RearDepths[i]) continue;
        float t = eyeDistance / (eyeDistance - centerDistance);
        float3 hitWorld = lerp(eyeWorld, centerWorld, t);
        float3 hitLocal = mul(_Stage1WorldToLocal[i], float4(hitWorld, 1)).xyz;
        float4 bounds = _Stage1BoundsMinMaxXZ[i];
        if (hitLocal.x >= bounds.x && hitLocal.x <= bounds.y &&
            hitLocal.z >= bounds.z && hitLocal.z <= bounds.w) return true;
    }
    return false;
}

#endif
