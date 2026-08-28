"""Stage 3: camera trajectory + sparse cloud via COLMAP on the perspective crops.

Pipeline: feature_extractor (fixed PINHOLE intrinsics, hull masks) ->
sequential_matcher -> mapper -> model_converter TXT -> ingest_model():
recover per-frame rig poses from registered views, scale to metres from the
average vehicle speed, write track/poses.json + track/colmap/ + QC plot.

stella_vslam (equirect VSLAM) remains the intended backend for the 12 km
batch; this COLMAP path is the no-Docker fallback per spec section 7.
"""
import argparse
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .colmap_export import (load_images_txt, load_points3d_full,
                            write_colmap_model)
from .io_utils import Project, ensure_dir, read_json, read_jsonl, write_json
from .trajectory import arc_length, quat_to_R, yaw_view_pose

YAW_PRIORITY = (90, -90, 45, -45, 135, -135, 0, 180)


def rig_pose_from_view(T_view, yaw_deg):
    a = math.radians(yaw_deg)
    Ry = np.array([[math.cos(a), 0, math.sin(a)],
                   [0, 1, 0],
                   [-math.sin(a), 0, math.cos(a)]])
    T = T_view.copy()
    T[:3, :3] = T_view[:3, :3] @ Ry.T
    return T


def _parse_view_name(name):
    stem, ytok = Path(name).stem.rsplit("_y", 1)
    return stem, int(ytok)


def drop_rig_outliers(views, max_dev=0.5):
    """All views of a frame are crops of one 360 photo, so they must share one
    optical centre. Views whose position deviates from their frame's median
    centre by more than max_dev metres are mis-registered -> dropped."""
    by_stem = {}
    for v in views:
        stem, _ = _parse_view_name(v["name"])
        by_stem.setdefault(stem, []).append(v)
    kept, dropped = [], []
    for stem, group in by_stem.items():
        centers = np.array([g["T_wc"][:3, 3] for g in group])
        med = np.median(centers, axis=0)
        for g, c in zip(group, centers):
            (kept if np.linalg.norm(c - med) <= max_dev else dropped).append(g)
    return kept, dropped


def rig_poses_from_views(views):
    """views: [{"name": frame_XXXX_y+090.jpg, "T_wc": 4x4}] -> {stem: T_rig},
    picking the highest-priority registered yaw per frame."""
    by_stem = {}
    for v in views:
        stem, yaw = _parse_view_name(v["name"])
        by_stem.setdefault(stem, {})[yaw] = v["T_wc"]
    out = {}
    for stem, yaws in by_stem.items():
        for y in YAW_PRIORITY:
            if y in yaws:
                out[stem] = rig_pose_from_view(yaws[y], y)
                break
    return out


def compute_scale(times, positions, speed_kmh):
    dist_model = arc_length(np.asarray(positions))[-1]
    dist_real = (speed_kmh / 3.6) * (times[-1] - times[0])
    return dist_real / dist_model


def prepare_colmap_masks(project, out_dir):
    """COLMAP expects mask filename = <image filename>.png (e.g. x.jpg.png)."""
    out = ensure_dir(out_dir)
    for m in project.views_masks_dir.glob("*.png"):
        shutil.copy2(m, out / (m.stem + ".jpg.png"))
    return out


def build_feature_cmd(exe, db, image_path, mask_path, intr):
    params = f"{intr['fx']},{intr['fy']},{intr['cx']},{intr['cy']}"
    return [str(exe), "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(image_path),
            "--ImageReader.mask_path", str(mask_path),
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params", params]


def build_matcher_cmd(exe, db, overlap=36):
    return [str(exe), "sequential_matcher",
            "--database_path", str(db),
            "--SequentialMatching.overlap", str(overlap)]


def build_mapper_cmd(exe, db, image_path, out_dir):
    return [str(exe), "mapper",
            "--database_path", str(db),
            "--image_path", str(image_path),
            "--output_path", str(out_dir),
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0"]


def build_converter_cmd(exe, model_dir, out_dir):
    return [str(exe), "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(out_dir),
            "--output_type", "TXT"]


def pick_largest_model(sparse_dir):
    best, best_n = None, -1
    for sub in sorted(Path(sparse_dir).iterdir()):
        img_bin = sub / "images.bin"
        if img_bin.exists():
            n = img_bin.stat().st_size
            if n > best_n:
                best, best_n = sub, n
    return best


def _qc_plot(positions, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(positions[:, 0], positions[:, 2], ".-", ms=3)
    ax1.set_title("top-down (x,z) [m]"); ax1.set_aspect("equal")
    s = arc_length(positions)
    ax2.plot(s, -positions[:, 1])  # y-down convention -> plot height as -y
    ax2.set_title("height vs arc-length [m]")
    fig.savefig(out_path, dpi=120); plt.close(fig)


def ingest_model(project, model_txt_dir, speed_kmh):
    p = project if isinstance(project, Project) else Project(project)
    model_txt_dir = Path(model_txt_dir)
    imgs = load_images_txt(model_txt_dir / "images.txt")
    views = []
    for im in imgs:
        qw, qx, qy, qz = im["q"]
        R_cw = quat_to_R(qx, qy, qz, qw)
        T = np.eye(4)
        T[:3, :3] = R_cw.T
        T[:3, 3] = -R_cw.T @ np.asarray(im["t"])
        views.append({"name": im["name"], "T_wc": T})

    rigs = rig_poses_from_views(views)
    t_by_stem = {Path(f["name"]).stem: f["t"] for f in read_jsonl(p.frames_meta)}
    stems = sorted(s for s in rigs if s in t_by_stem)
    times = [t_by_stem[s] for s in stems]
    positions = np.array([rigs[s][:3, 3] for s in stems])
    scale = compute_scale(times, positions, speed_kmh)

    # rig-consistency validation (threshold is metric, poses still unscaled)
    views, bad = drop_rig_outliers(views, max_dev=0.5 / scale)
    if bad:
        print(f"rig validation: dropped {len(bad)} mis-registered views: "
              + ", ".join(sorted(v['name'] for v in bad)[:8])
              + (" ..." if len(bad) > 8 else ""))
        rigs = rig_poses_from_views(views)
        stems = sorted(s for s in rigs if s in t_by_stem)
        times = [t_by_stem[s] for s in stems]
        positions = np.array([rigs[s][:3, 3] for s in stems])
        scale = compute_scale(times, positions, speed_kmh)

    for v in views:
        v["T_wc"] = v["T_wc"].copy()
        v["T_wc"][:3, 3] *= scale
    points = load_points3d_full(model_txt_dir / "points3D.txt")
    if points is not None:
        points[:, :3] *= scale
        # radial filter: far low-parallax points (sky remnants, distant haze)
        from scipy.spatial import cKDTree
        d, _ = cKDTree(positions * scale).query(points[:, :3])
        n_far = int((d > 60.0).sum())
        points = points[d <= 60.0]
        if n_far:
            print(f"radial filter: dropped {n_far:,} points farther than 60 m")

    meta = read_json(p.views_meta)
    write_colmap_model(p.colmap_dir, meta, views, points)

    positions_scaled = positions * scale
    s_vals = arc_length(positions_scaled)
    write_json(p.poses_json, {"scale": scale, "frames": [
        {"name": stem + ".jpg", "t": t_by_stem[stem], "s": float(sv),
         "T_wc": [float(x) for x in _scaled(rigs[stem], scale).reshape(-1)]}
        for stem, sv in zip(stems, s_vals)]})
    _qc_plot(positions_scaled, p.qc_trajectory)

    n_total = len(list(p.views_dir.glob("*.jpg")))
    print(f"registered {len(views)}/{n_total} views, {len(stems)} rig frames, "
          f"scale={scale:.4f}, length={s_vals[-1]:.1f} m, "
          f"points={0 if points is None else len(points)}")


def _scaled(T, s):
    T2 = T.copy()
    T2[:3, 3] *= s
    return T2


def run_track(project_dir, colmap_exe, speed_kmh, overlap=36):
    p = Project(project_dir)
    work = ensure_dir(p.track_dir / "colmap_work")
    db = work / "database.db"
    masks = prepare_colmap_masks(p, work / "masks")
    sparse = ensure_dir(work / "sparse")

    for cmd in (build_feature_cmd(colmap_exe, db, p.views_dir, masks,
                                  read_json(p.views_meta)),
                build_matcher_cmd(colmap_exe, db, overlap),
                build_mapper_cmd(colmap_exe, db, p.views_dir, sparse)):
        print(">>", " ".join(str(c) for c in cmd[:2]))
        subprocess.run([str(c) for c in cmd], check=True)

    model = pick_largest_model(sparse)
    if model is None:
        raise SystemExit("mapper produced no model")
    txt = ensure_dir(work / "sparse_txt")
    subprocess.run([str(c) for c in build_converter_cmd(colmap_exe, model, txt)],
                   check=True)
    ingest_model(p, txt, speed_kmh)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--colmap", default=r"C:\Work\tools\colmap\bin\colmap.exe")
    ap.add_argument("--speed-kmh", type=float, required=True)
    ap.add_argument("--overlap", type=int, default=36)
    ap.add_argument("--ingest-only", action="store_true",
                    help="skip COLMAP, re-ingest existing sparse_txt model")
    a = ap.parse_args()
    if a.ingest_only:
        p = Project(a.project)
        ingest_model(p, p.track_dir / "colmap_work" / "sparse_txt", a.speed_kmh)
    else:
        run_track(a.project, a.colmap, a.speed_kmh, a.overlap)


if __name__ == "__main__":
    main()
