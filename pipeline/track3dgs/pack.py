"""Stage 7: gravity-align tiles, emit the Unity tile manifest + merged demo PLY.

Alignment: the camera is rigidly mounted and roughly level, so the average
camera 'up' (-Y column of the rig rotations) is a robust world-up estimate.
We rotate the world so that up maps to -Y (our frames stay y-down; Unity's
importer flips to Y-up with its usual (x,-y,z) convention) and translate the
trajectory start to the origin.

SH bands (f_rest_*) are stripped from packed tiles by default: rotating SH
correctly needs Wigner matrices, and the Quest target wants SH0 anyway
(3x smaller). The full-SH unrotated cell PLY stays in train/export/.
"""
import argparse
import re

import numpy as np
from plyfile import PlyData, PlyElement

from .io_utils import Project, ensure_dir, read_json, write_json
from .trajectory import R_to_quat


def _rotation_between(a, b):
    """Minimal rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def ground_up_ransac(points, cam_positions, iters=500, tol=0.3, seed=0):
    """Up = RANSAC ground-plane normal of near-track points, signed toward
    the cameras (which sit above the road)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(cam_positions).query(points)
    near = points[d < 20.0]
    if len(near) < 100:
        return None
    rng = np.random.default_rng(seed)
    best_n, best_a, best_cnt = None, None, 0
    for _ in range(iters):
        a, b, c = near[rng.choice(len(near), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        cnt = int((np.abs((near - a) @ n) < tol).sum())
        if cnt > best_cnt:
            best_cnt, best_n, best_a = cnt, n, a
    if best_n is None:
        return None
    if np.mean((cam_positions - best_a) @ best_n) < 0:
        best_n = -best_n
    return best_n


def estimate_alignment(frames, points=None):
    """Return (R, t0). R maps world-up to +Y (viewer/Unity convention) and
    the mean travel direction into the +Z half-space; t0 is the trajectory
    start. Up comes from the RANSAC ground plane when points are given
    (geometry-based, robust to a tilted camera mount), else from the mean
    camera up-axis."""
    Ts = [np.array(f["T_wc"]).reshape(4, 4) for f in frames]
    cam_positions = np.array([T[:3, 3] for T in Ts])
    cam_up = -np.mean([T[:3, 1] for T in Ts], axis=0)
    cam_up /= np.linalg.norm(cam_up)

    up = None
    if points is not None:
        up = ground_up_ransac(points, cam_positions)
    if up is None:
        up = cam_up
    elif np.dot(up, cam_up) < 0:
        up = -up

    R1 = _rotation_between(up, np.array([0.0, 1.0, 0.0]))
    travel = R1 @ (cam_positions[-1] - cam_positions[0])
    travel[1] = 0.0
    if np.linalg.norm(travel) > 1e-9:
        travel /= np.linalg.norm(travel)
        R2 = _rotation_between(travel, np.array([0.0, 0.0, 1.0]))
    else:
        R2 = np.eye(3)
    R = R2 @ R1
    t0 = Ts[0][:3, 3].copy()
    return R, t0


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def transform_splats(v, R, t0, strip_sh=True):
    """Apply p' = R @ (p - t0) to positions and compose R into the splat
    quaternions (rot_0..3, wxyz). Optionally drop SH band fields."""
    names = [n for n in v.dtype.names
             if not (strip_sh and n.startswith("f_rest_"))]
    out = np.zeros(len(v), dtype=[(n, v.dtype[n]) for n in names])
    for n in names:
        out[n] = v[n]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64) - t0
    xyz = xyz @ R.T
    out["x"], out["y"], out["z"] = (xyz[:, 0].astype(np.float32),
                                    xyz[:, 1].astype(np.float32),
                                    xyz[:, 2].astype(np.float32))

    qR = np.array(R_to_quat(R))
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                     axis=1).astype(np.float64)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    quats = quats / norms
    rotated = np.stack([_quat_mul(qR, q) for q in quats])
    for i, n in enumerate(["rot_0", "rot_1", "rot_2", "rot_3"]):
        out[n] = rotated[:, i].astype(np.float32)
    return out


def tile_record(ply_path, tile_len, vertex=None):
    tid = int(re.search(r"tile_(\d+)", ply_path.name).group(1))
    v = vertex if vertex is not None else PlyData.read(str(ply_path))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    return {"id": tid, "file": ply_path.name,
            "s_start": tid * tile_len, "s_end": (tid + 1) * tile_len,
            "num_splats": int(len(v)),
            "bounds": {"min": [float(x) for x in xyz.min(0)],
                       "max": [float(x) for x in xyz.max(0)]}}


def euler_to_R(x_deg, y_deg, z_deg):
    """PlayCanvas/SuperSplat Euler convention: R = Rz @ Ry @ Rx (XYZ intrinsic)."""
    x, y, z = np.radians([x_deg, y_deg, z_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def run_pack(project_dir, merge=False, strip_sh=True, euler=None):
    p = Project(project_dir)
    cj = read_json(p.cells_json)
    tile_len = cj["tile_length"]
    frames = read_json(p.poses_json)["frames"]
    from .colmap_export import load_points3d_txt
    pts_file = p.colmap_dir / "points3D.txt"
    points = load_points3d_txt(pts_file) if pts_file.exists() else None
    R, t0 = estimate_alignment(frames,
                               points[:, :3] if points is not None else None)
    if euler is not None:
        # viewer-measured correction (e.g. from SuperSplat's transform panel)
        # baked on top of the automatic alignment
        R = euler_to_R(*euler) @ R
    if points is not None:
        cams = np.array([np.array(f["T_wc"]).reshape(4, 4)[:3, 3]
                         for f in frames])
        n = ground_up_ransac(points[:, :3], cams)
        if n is not None:
            resid = np.degrees(np.arccos(np.clip((R @ n)[1], -1, 1)))
            print(f"ground-plane tilt after alignment: {resid:.2f} deg")

    packed_dir = ensure_dir(p.tiles_dir / "packed")
    tiles, merged_parts = [], []
    for f in sorted(p.tiles_dir.glob("tile_*.ply")):
        v = PlyData.read(str(f))["vertex"].data
        tv = transform_splats(v, R, t0, strip_sh)
        out_path = packed_dir / f.name
        PlyData([PlyElement.describe(tv, "vertex")]).write(str(out_path))
        tiles.append(tile_record(out_path, tile_len, vertex=tv))
        merged_parts.append(tv)
        print(f"{f.name}: {len(tv):,} splats packed")

    traj = []
    for fr in frames:
        pos = np.array(fr["T_wc"]).reshape(4, 4)[:3, 3]
        pos = R @ (pos - t0)
        traj.append({"s": fr["s"], "pos": [float(x) for x in pos]})

    write_json(p.manifest_json, {
        "version": 1, "tile_length": tile_len, "total_s": cj["total_s"],
        "coordinate_convention": "right-handed, y-down (COLMAP); Unity import: (x,-y,z)",
        "sh_bands_stripped": bool(strip_sh),
        "alignment": {"rotation": [float(x) for x in R.reshape(-1)],
                      "translation": [float(x) for x in t0]},
        "trajectory": traj, "tiles": tiles})
    print(f"manifest: {len(tiles)} tiles, "
          f"{sum(t['num_splats'] for t in tiles):,} splats total")

    if merge and merged_parts:
        merged = np.concatenate(merged_parts)
        ensure_dir(p.demo_dir)
        PlyData([PlyElement.describe(merged, "vertex")]).write(
            str(p.demo_dir / "section_merged.ply"))
        print(f"merged: {len(merged):,} splats -> demo/section_merged.ply")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--keep-sh", action="store_true")
    ap.add_argument("--euler", default=None,
                    help="viewer rotation to bake, SuperSplat-style 'x,y,z' degrees")
    a = ap.parse_args()
    euler = tuple(float(v) for v in a.euler.split(",")) if a.euler else None
    run_pack(a.project, a.merge, strip_sh=not a.keep_sh, euler=euler)


if __name__ == "__main__":
    main()
