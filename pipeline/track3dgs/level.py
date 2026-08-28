"""Stage 4: level a section's COLMAP frame using the mount calibration.

The 360 camera is rigidly mounted, so its axes have a fixed relationship to
gravity. That relationship was measured once from the user's manual SuperSplat
leveling of section01 (leveling_calibration.json) and stored as the mount key:
where the camera up-axis and travel direction point in a correctly-leveled
world. Any section is leveled by rotating its frame so its own camera axes
match the key. Origin moves to the trajectory start.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .colmap_export import (load_images_txt, load_points3d_txt,
                            write_colmap_model)
from .io_utils import Project, read_json, write_json
from .trajectory import quat_to_R

MOUNT_FILE = Path(__file__).parent.parent / "mount_calibration.json"


def _cam_axes(frames):
    Ts = [np.array(f["T_wc"]).reshape(4, 4) for f in frames]
    up = -np.mean([T[:3, 1] for T in Ts], axis=0)
    up /= np.linalg.norm(up)
    travel = Ts[-1][:3, 3] - Ts[0][:3, 3]
    travel /= np.linalg.norm(travel)
    return up, travel


def _triad(up, travel):
    e1 = up / np.linalg.norm(up)
    e2 = travel - np.dot(travel, e1) * e1
    e2 /= np.linalg.norm(e2)
    return np.stack([e1, e2, np.cross(e1, e2)], axis=1)


def make_mount_key(frames_oldframe, R_level):
    """One-off: push the old frame's camera axes through the user's measured
    leveling rotation -> the axes' true directions in a leveled world."""
    up, travel = _cam_axes(frames_oldframe)
    return {"cam_up": (R_level @ up).tolist(),
            "travel": (R_level @ travel).tolist()}


def orientation_from_key(frames, key):
    """Rotation that puts this section's camera axes onto the calibrated key."""
    up_s, travel_s = _cam_axes(frames)
    R = _triad(np.array(key["cam_up"]), np.array(key["travel"])) @ \
        _triad(up_s, travel_s).T
    t0 = np.array(frames[0]["T_wc"]).reshape(4, 4)[:3, 3]
    return R, t0


def _apply(T, R, t0):
    out = T.copy()
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = R @ (T[:3, 3] - t0)
    return out


def run_level(project_dir, mount_file=MOUNT_FILE):
    p = Project(project_dir)
    key = json.loads(Path(mount_file).read_text())
    poses = read_json(p.poses_json)
    frames = poses["frames"]
    R, t0 = orientation_from_key(frames, key)

    for f in frames:
        T = _apply(np.array(f["T_wc"]).reshape(4, 4), R, t0)
        f["T_wc"] = [float(v) for v in T.reshape(-1)]
    poses["leveled"] = True
    write_json(p.poses_json, poses)

    meta = read_json(p.views_meta)
    imgs = load_images_txt(p.colmap_dir / "images.txt")
    views = []
    for im in imgs:
        qw, qx, qy, qz = im["q"]
        R_cw = quat_to_R(qx, qy, qz, qw)
        T = np.eye(4)
        T[:3, :3] = R_cw.T
        T[:3, 3] = -R_cw.T @ np.asarray(im["t"])
        views.append({"name": im["name"], "T_wc": _apply(T, R, t0)})
    points = load_points3d_txt(p.colmap_dir / "points3D.txt")
    if points is not None:
        points[:, :3] = (points[:, :3] - t0) @ R.T
    write_colmap_model(p.colmap_dir, meta, views, points)

    write_json(p.track_dir / "level_transform.json",
               {"rotation": R.tolist(), "origin": t0.tolist(),
                "mount_file": str(mount_file)})

    up_after, travel_after = _cam_axes(frames)
    err_up = np.degrees(np.arccos(np.clip(
        np.dot(up_after, key["cam_up"]), -1, 1)))
    print(f"leveled: cam-up residual {err_up:.2f} deg, "
          f"trajectory starts at origin")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--mount", default=str(MOUNT_FILE))
    a = ap.parse_args()
    run_level(a.project, a.mount)


if __name__ == "__main__":
    main()
