"""Stage 6: slice a trained cell PLY into non-overlapping 10 m tiles + prune.

Every Gaussian is assigned to exactly one tile by its arc-length position
along the trajectory; pad-zone Gaussians (outside the cell core) are dropped
here because the neighbouring cell owns them. Pruning removes Gaussians far
from the viewer tube (sky plugs, far fluff) and near-invisible ones.
"""
import argparse

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from .io_utils import Project, ensure_dir, read_json


def gaussian_s_values(xyz, traj_pts, traj_s):
    tree = cKDTree(traj_pts)
    dist, idx = tree.query(xyz)
    return traj_s[idx], dist


def assign_tiles(s, core_lo, core_hi, tile_len):
    t = (s // tile_len).astype(int)
    t[(s < core_lo) | (s >= core_hi)] = -1
    return t


def prune_mask(dist, opacity_raw, max_dist, min_opacity):
    op = 1.0 / (1.0 + np.exp(-opacity_raw))
    return (dist <= max_dist) & (op >= min_opacity)


def support_mask(xyz, support_points, max_support_dist):
    """Keep only Gaussians near photo-consistent geometry (COLMAP points).
    Sky shells, underground mirror-fluff and end plugs have no support."""
    tree = cKDTree(support_points)
    d, _ = tree.query(xyz)
    return d <= max_support_dist


def run_slice(project_dir, cell_id, max_dist=60.0, min_opacity=0.005,
              max_support_dist=2.5):
    from .colmap_export import load_points3d_txt
    from .trajectory import resample_polyline
    p = Project(project_dir)
    cj = read_json(p.cells_json)
    cell = next(c for c in cj["cells"] if c["id"] == cell_id)
    tile_len = cj["tile_length"]

    frames = read_json(p.poses_json)["frames"]
    positions = np.array([np.array(f["T_wc"]).reshape(4, 4)[:3, 3] for f in frames])
    traj_pts, traj_s = resample_polyline(positions, step=0.5)

    ply = PlyData.read(str(p.export_dir / f"cell_{cell_id:03d}.ply"))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(float)
    s, dist = gaussian_s_values(xyz, traj_pts, traj_s)
    keep = prune_mask(dist, v["opacity"].astype(float), max_dist, min_opacity)
    if max_support_dist > 0:
        pts_file = p.colmap_dir / "points3D.txt"
        if pts_file.exists():
            support = load_points3d_txt(pts_file)
            n_before = int(keep.sum())
            keep &= support_mask(xyz, support[:, :3], max_support_dist)
            print(f"support pruning: {n_before - int(keep.sum()):,} unsupported "
                  f"splats dropped")
    tiles = assign_tiles(s, cell["s_core"][0], cell["s_core"][1], tile_len)

    ensure_dir(p.tiles_dir)
    n_pad = int((tiles == -1).sum())
    for tid in sorted(set(tiles[keep & (tiles >= 0)])):
        sel = keep & (tiles == tid)
        el = PlyElement.describe(v[sel], "vertex")
        PlyData([el]).write(str(p.tiles_dir / f"tile_{tid:04d}.ply"))
        print(f"tile_{tid:04d}: {int(sel.sum()):,} splats")
    print(f"pruned {int((~keep).sum()):,} / pad-dropped {n_pad:,} of {len(v):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--max-dist", type=float, default=60.0)
    ap.add_argument("--min-opacity", type=float, default=0.005)
    ap.add_argument("--max-support-dist", type=float, default=2.5,
                    help="drop splats farther than this from any COLMAP point; 0 disables")
    a = ap.parse_args()
    run_slice(a.project, a.cell, a.max_dist, a.min_opacity, a.max_support_dist)


if __name__ == "__main__":
    main()
