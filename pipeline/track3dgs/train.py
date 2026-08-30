"""Stage 5: per-cell Splatfacto training in the fixed global frame.

Runs ns-train from the dedicated training venv (.venv-train) with pose
normalization disabled so the exported PLY stays in global metric
coordinates, then ns-export, then verifies the export actually landed in the
global frame (guard against normalization leaking back in).
"""
import argparse
import shutil
import subprocess

import numpy as np
from plyfile import PlyData, PlyElement

from .io_utils import Project, ensure_dir

# nerfstudio's colmap loader bakes a COLMAP->OpenGL rotation (recorded in the
# run's dataparser_transforms.json) that ns-export gaussian-splat does NOT
# invert; this is its exact inverse, applied to every export so the PLY lands
# back in our global frame.
NS_EXPORT_FIX = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def build_train_cmd(project, cell_id, iters):
    cell_dir = project.cells_dir / f"cell_{cell_id:03d}"
    return ["ns-train", "splatfacto",
            "--data", str(cell_dir),
            "--output-dir", str(project.train_dir),
            "--experiment-name", f"cell_{cell_id:03d}",
            "--timestamp", "run",
            "--max-num-iterations", str(iters),
            "--viewer.quit-on-train-completion", "True",
            # nerfstudio 1.1.5 exposes no MCMC strategy / hard splat cap
            # (verified via ns-train --help); the Quest splat budget is
            # enforced later in slice/pack pruning per spec section 3.
            "colmap",
            "--colmap-path", "colmap",
            "--images-path", str(project.views_dir.resolve()),
            "--masks-path", str(project.views_masks_dir.resolve()),
            "--center-method", "none",
            "--orientation-method", "none",
            "--auto-scale-poses", "False",
            "--load-3D-points", "True"]


def build_export_cmd(config_yml, out_dir):
    return ["ns-export", "gaussian-splat",
            "--load-config", str(config_yml),
            "--output-dir", str(out_dir)]


def check_alignment(exported_ply, cell_colmap_dir, max_offset=1.5,
                    n_sample=2000, seed=0):
    """Frame-leak guard: median nearest-neighbour distance from sampled
    splats to the cell's COLMAP points. Both lie on real surfaces, so in the
    correct frame the median is centimetres; any leaked rotation/flip/scale
    puts it at metres. Robust to how densely each cloud samples the scene
    (median cloud offsets are not)."""
    from scipy.spatial import cKDTree
    v = PlyData.read(str(exported_ply))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(float)
    pts = []
    for line in (cell_colmap_dir / "points3D.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split()
            pts.append([float(f[1]), float(f[2]), float(f[3])])
    pts = np.array(pts)
    rng = np.random.default_rng(seed)
    sample = xyz[rng.choice(len(xyz), min(n_sample, len(xyz)), replace=False)]
    d, _ = cKDTree(pts).query(sample)
    offset = float(np.median(d))
    return {"offset_m": offset, "ok": offset < max_offset}


def run_train(project_dir, cell_id, iters=30000, dry_run=False,
              export_only=False):
    p = Project(project_dir)
    cmd = build_train_cmd(p, cell_id, iters)
    cfg = p.train_dir / f"cell_{cell_id:03d}" / "splatfacto" / "run" / "config.yml"
    exp = build_export_cmd(cfg, p.export_dir / f"cell_{cell_id:03d}")
    if dry_run:
        print(" ".join(str(c) for c in cmd))
        print(" ".join(str(c) for c in exp))
        return
    if not export_only:
        subprocess.run([str(c) for c in cmd], check=True)
    subprocess.run([str(c) for c in exp], check=True)
    src = p.export_dir / f"cell_{cell_id:03d}" / "splat.ply"
    dst = p.export_dir / f"cell_{cell_id:03d}.ply"
    ensure_dir(p.export_dir)
    shutil.move(str(src), str(dst))
    from .pack import transform_splats
    import gc
    ply = PlyData.read(str(dst))
    fixed = transform_splats(ply["vertex"].data, NS_EXPORT_FIX,
                             np.zeros(3), strip_sh=False)
    # plyfile memory-maps on read; Windows cannot truncate a mapped file,
    # so release the mapping before writing back to the same path
    del ply
    gc.collect()
    PlyData([PlyElement.describe(fixed, "vertex")]).write(str(dst))
    res = check_alignment(dst, p.cells_dir / f"cell_{cell_id:03d}" / "colmap")
    print(f"alignment offset {res['offset_m']:.2f} m -> "
          f"{'OK' if res['ok'] else 'FAILED (normalization leaked!)'}")
    if not res["ok"]:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--export-only", action="store_true",
                    help="skip training; export + alignment-check an existing run")
    a = ap.parse_args()
    run_train(a.project, a.cell, a.iters, a.dry_run, a.export_only)


if __name__ == "__main__":
    main()
