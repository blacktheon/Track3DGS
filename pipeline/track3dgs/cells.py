"""Stage 4: arc-length cell planning + per-cell COLMAP subsets (global frame)."""
import argparse

import numpy as np

from .colmap_export import load_images_txt, load_points3d_txt, write_colmap_model
from .io_utils import Project, read_json, write_json
from .trajectory import quat_to_R


def plan_cells(total_s, cell=50.0, pad=10.0):
    n = max(1, int(total_s // cell))
    if total_s - n * cell >= cell / 2:
        n += 1
    cells = []
    for i in range(n):
        s0, s1 = i * cell, min((i + 1) * cell, total_s)
        if i == n - 1:
            s1 = total_s
        cells.append({"id": i, "s_core": [s0, s1],
                      "s_full": [max(0.0, s0 - pad), min(total_s, s1 + pad)]})
    return cells


def invert_wc(q, t):
    qw, qx, qy, qz = q
    R_cw = quat_to_R(qx, qy, qz, qw)
    T = np.eye(4)
    T[:3, :3] = R_cw.T
    T[:3, 3] = -R_cw.T @ np.asarray(t, dtype=float)
    return T


def run_cells(project_dir, cell=50.0, pad=10.0, tile=10.0, bbox_margin=20.0):
    p = Project(project_dir)
    poses = read_json(p.poses_json)
    frames = poses["frames"]
    total_s = frames[-1]["s"]
    cells = plan_cells(total_s, cell, pad)

    meta = read_json(p.views_meta)
    all_imgs = load_images_txt(p.colmap_dir / "images.txt")
    points = load_points3d_txt(p.colmap_dir / "points3D.txt")

    for c in cells:
        lo, hi = c["s_full"]
        cframes = [f for f in frames if lo <= f["s"] <= hi]
        c["frames"] = [f["name"] for f in cframes]
        stems = {f["name"].rsplit(".", 1)[0] for f in cframes}
        cimgs = [{"name": im["name"], "T_wc": invert_wc(im["q"], im["t"])}
                 for im in all_imgs
                 if im["name"].rsplit("_y", 1)[0] in stems]
        cpts = None
        if points is not None and cframes:
            centers = np.array([np.array(f["T_wc"]).reshape(4, 4)[:3, 3]
                                for f in cframes])
            bb_lo = centers.min(0) - bbox_margin
            bb_hi = centers.max(0) + bbox_margin
            inside = np.all((points[:, :3] >= bb_lo) & (points[:, :3] <= bb_hi),
                            axis=1)
            cpts = points[inside] if inside.any() else None
        write_colmap_model(p.cells_dir / f"cell_{c['id']:03d}" / "colmap",
                           meta, cimgs, cpts)
        n_pts = 0 if cpts is None else len(cpts)
        print(f"cell_{c['id']:03d}: s_core={c['s_core']}, "
              f"{len(cframes)} frames, {len(cimgs)} views, {n_pts} points")

    write_json(p.cells_json, {"cell_length": cell, "pad": pad,
                              "tile_length": tile, "total_s": total_s,
                              "cells": cells})
    print(f"{len(cells)} cells over {total_s:.1f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--cell", type=float, default=50.0)
    ap.add_argument("--pad", type=float, default=10.0)
    ap.add_argument("--tile", type=float, default=10.0)
    a = ap.parse_args()
    run_cells(a.project, a.cell, a.pad, a.tile)


if __name__ == "__main__":
    main()
