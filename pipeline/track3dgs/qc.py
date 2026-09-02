"""Per-section quality report: registration, alignment guard, splat counts,
prune ratio, masked PSNR. Prints a summary block and appends one row to
data/Export/qc_log.csv. Run inside cuda_env (PSNR renders with gsplat).
"""
import argparse
import csv
from pathlib import Path

from plyfile import PlyData

from .colmap_export import load_images_txt
from .evalpsnr import run_eval
from .io_utils import Project, read_json
from .train import check_alignment


def run_qc(project_dir, n_views=8, csv_path=r"..\data\Export\qc_log.csv"):
    p = Project(project_dir)
    name = Path(project_dir).name

    n_views_total = len(list(p.views_dir.glob("*.jpg")))
    registered = len(load_images_txt(p.colmap_dir / "images.txt"))
    poses = read_json(p.poses_json)
    n_frames = len(poses["frames"])
    length = poses["frames"][-1]["s"]

    raw_ply = p.export_dir / "cell_000.ply"
    final_ply = p.export_dir / "cell_000_skypruned.ply"
    n_raw = len(PlyData.read(str(raw_ply))["vertex"].data)
    n_final = len(PlyData.read(str(final_ply))["vertex"].data)

    guard = check_alignment(final_ply, p.cells_dir / "cell_000" / "colmap")
    psnr = run_eval(p.root, final_ply, n_views=n_views)

    row = {
        "section": name,
        "views_registered": f"{registered}/{n_views_total}",
        "rig_frames": n_frames,
        "length_m": round(length, 1),
        "guard_m": round(guard["offset_m"], 3),
        "guard_ok": guard["ok"],
        "splats_trained": n_raw,
        "splats_final": n_final,
        "pruned_pct": round(100 * (1 - n_final / n_raw), 1),
        "psnr_db": round(psnr, 2),
    }
    print("=" * 20 + f" QC {name} " + "=" * 20)
    for k, v in row.items():
        print(f"  {k:16s} {v}")
    if not guard["ok"]:
        print("  *** ALIGNMENT GUARD FAILED - inspect before using ***")

    out = Path(csv_path)
    out.parent.mkdir(exist_ok=True)
    exists = out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--n-views", type=int, default=8)
    a = ap.parse_args()
    run_qc(a.project, a.n_views)


if __name__ == "__main__":
    main()
