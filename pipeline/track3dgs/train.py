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
from plyfile import PlyData

from .io_utils import Project, ensure_dir


def build_train_cmd(project, cell_id, iters, cap):
    cell_dir = project.cells_dir / f"cell_{cell_id:03d}"
    return ["ns-train", "splatfacto",
            "--data", str(cell_dir),
            "--output-dir", str(project.train_dir),
            "--experiment-name", f"cell_{cell_id:03d}",
            "--timestamp", "run",
            "--max-num-iterations", str(iters),
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.model.strategy", "mcmc",
            "--pipeline.model.max-gs-num", str(cap),
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


def check_alignment(exported_ply, cell_colmap_dir, max_offset=2.0):
    v = PlyData.read(str(exported_ply))["vertex"]
    centroid = np.array([np.median(v["x"]), np.median(v["y"]), np.median(v["z"])])
    pts = []
    for line in (cell_colmap_dir / "points3D.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split()
            pts.append([float(f[1]), float(f[2]), float(f[3])])
    ref = np.median(np.array(pts), axis=0)
    offset = float(np.linalg.norm(centroid - ref))
    return {"offset_m": offset, "ok": offset < max_offset}


def run_train(project_dir, cell_id, iters=30000, cap=2_500_000, dry_run=False):
    p = Project(project_dir)
    cmd = build_train_cmd(p, cell_id, iters, cap)
    cfg = p.train_dir / f"cell_{cell_id:03d}" / "splatfacto" / "run" / "config.yml"
    exp = build_export_cmd(cfg, p.export_dir / f"cell_{cell_id:03d}")
    if dry_run:
        print(" ".join(str(c) for c in cmd))
        print(" ".join(str(c) for c in exp))
        return
    subprocess.run([str(c) for c in cmd], check=True)
    subprocess.run([str(c) for c in exp], check=True)
    src = p.export_dir / f"cell_{cell_id:03d}" / "splat.ply"
    dst = p.export_dir / f"cell_{cell_id:03d}.ply"
    ensure_dir(p.export_dir)
    shutil.move(str(src), str(dst))
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
    ap.add_argument("--cap", type=int, default=2_500_000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run_train(a.project, a.cell, a.iters, a.cap, a.dry_run)


if __name__ == "__main__":
    main()
