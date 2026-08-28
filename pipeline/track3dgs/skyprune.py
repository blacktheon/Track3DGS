"""Mask-projection sky pruner: lift the 2D sky masks into 3D.

For every Gaussian, project its centre into each frame's equirect sky mask
(via the rig pose - the full sphere is visible, no bounds cases). A Gaussian
whose direction lands on masked sky from most nearby frames IS sky, whatever
its colour. Geometric, not chromatic: blue splats on real treetops survive
(they project onto unmasked canopy).

Reads train/export/cell_XXX.ply, writes cell_XXX_skypruned.ply + stats.
"""
import argparse

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from .io_utils import Project, read_json


def equirect_uv(dirs_cam, W, H):
    """Camera-frame direction vectors (x-right, y-down, z-forward) ->
    equirect pixel coordinates."""
    d = dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    lon = np.arctan2(d[:, 0], d[:, 2])            # 0 forward, +right
    lat = np.arcsin(np.clip(-d[:, 1], -1, 1))     # + up (y-down convention)
    u = (lon / (2 * np.pi) + 0.5) * W
    v = (0.5 - lat / np.pi) * H
    return u, v


def sky_fractions(centers, frames, masks, max_range=50.0):
    """Fraction of nearby frames from which each centre projects onto sky.
    masks: {frame_stem: HxW uint8 keep-mask (0 = sky)}."""
    n = len(centers)
    n_seen = np.zeros(n)
    n_sky = np.zeros(n)
    for f in frames:
        stem = f["name"].rsplit(".", 1)[0]
        if stem not in masks:
            continue
        mask = masks[stem]
        H, W = mask.shape
        T = np.array(f["T_wc"]).reshape(4, 4)
        rel = centers - T[:3, 3]
        dist = np.linalg.norm(rel, axis=1)
        near = dist <= max_range
        if not near.any():
            continue
        dirs_cam = rel[near] @ T[:3, :3]          # R_wc.T @ rel, row form
        u, v = equirect_uv(dirs_cam, W, H)
        ui = np.clip(u.astype(int), 0, W - 1)
        vi = np.clip(v.astype(int), 0, H - 1)
        hit_sky = mask[vi, ui] == 0
        n_seen[near] += 1
        n_sky[near] += hit_sky
    frac = np.zeros(n)
    ok = n_seen > 0
    frac[ok] = n_sky[ok] / n_seen[ok]
    return frac


SH0 = 0.28209479177387814


def splat_rgb(v):
    """SH0 coefficients -> approximate base color in [0,1]."""
    return np.clip(np.stack([0.5 + SH0 * v[f"f_dc_{i}"] for i in range(3)],
                            axis=1), 0, 1)


def sky_colored(rgb, min_blue=0.55, blue_margin=0.06, bright=0.85):
    """Sky-like base colour: distinctly blue, or blown-out white."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    blue = (b > min_blue) & (b > g + blue_margin) & (g > r)
    white = (r > bright) & (g > bright) & (b > bright)
    return blue | white


def run_skyprune(project_dir, cell_id=0, threshold=0.6, max_range=50.0,
                 mask_width=960, color_assist=True, canopy_height=2.0,
                 color_frac=0.12):
    p = Project(project_dir)
    src = p.export_dir / f"cell_{cell_id:03d}.ply"
    ply = PlyData.read(str(src))
    v = ply["vertex"].data
    centers = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(float)

    frames = read_json(p.poses_json)["frames"]
    masks = {}
    for f in tqdm(frames, desc="loading sky masks"):
        stem = f["name"].rsplit(".", 1)[0]
        m = cv2.imread(str(p.root / "sky_masks" / (stem + ".png")),
                       cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        masks[stem] = cv2.resize(m, (mask_width, mask_width // 2),
                                 interpolation=cv2.INTER_NEAREST)

    frac = sky_fractions(centers, frames, masks, max_range)
    prune = frac >= threshold
    n_geo = int(prune.sum())

    n_col = 0
    if color_assist:
        import json
        from .level import MOUNT_FILE
        key = json.loads(MOUNT_FILE.read_text())
        up = np.array(key["cam_up"])
        cam_pos = np.array([np.array(f["T_wc"]).reshape(4, 4)[:3, 3]
                            for f in frames])
        height = (centers - cam_pos.mean(0)) @ up
        glitter = (sky_colored(splat_rgb(v)) & (frac >= color_frac)
                   & (height > canopy_height) & ~prune)
        n_col = int(glitter.sum())
        prune |= glitter

    keep = ~prune
    out = p.export_dir / f"cell_{cell_id:03d}_skypruned.ply"
    kept = np.asarray(v[keep])
    el = PlyElement.describe(kept, "vertex")
    PlyData([el]).write(str(out))
    print(f"sky-pruned {int(prune.sum()):,} of {len(v):,} splats "
          f"({100 * prune.mean():.1f}%): {n_geo:,} geometric (dome) "
          f"+ {n_col:,} colour-assisted (canopy glitter) -> {out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--cell", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--max-range", type=float, default=50.0)
    a = ap.parse_args()
    run_skyprune(a.project, a.cell, a.threshold, a.max_range)


if __name__ == "__main__":
    main()
