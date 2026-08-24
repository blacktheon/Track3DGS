import numpy as np
import cv2
from track3dgs.io_utils import Project, ensure_dir, write_jsonl, write_json, read_json
from track3dgs.trajectory import yaw_view_pose
from track3dgs.colmap_export import write_colmap_model, load_images_txt
from track3dgs.track import (rig_pose_from_view, rig_poses_from_views,
                             compute_scale, prepare_colmap_masks,
                             build_feature_cmd, ingest_model)


def _rig_pose(x, heading_deg=0.0):
    T = yaw_view_pose(np.eye(4), heading_deg)   # reuse Ry for heading
    T[:3, 3] = [x, 0.0, 0.0]
    return T


def test_rig_pose_from_view_roundtrip():
    T_rig = _rig_pose(5.0, heading_deg=30.0)
    for yaw in (-135, -45, 90):
        T_view = yaw_view_pose(T_rig, yaw)
        back = rig_pose_from_view(T_view, yaw)
        assert np.allclose(back, T_rig, atol=1e-9)


def test_rig_poses_from_views_prefers_side_view():
    T_rig = _rig_pose(2.0)
    views = [{"name": "frame_000010_y+045.jpg", "T_wc": yaw_view_pose(T_rig, 45)},
             {"name": "frame_000010_y+090.jpg", "T_wc": yaw_view_pose(T_rig, 90)}]
    rigs = rig_poses_from_views(views)
    assert set(rigs) == {"frame_000010"}
    assert np.allclose(rigs["frame_000010"], T_rig, atol=1e-9)


def test_compute_scale():
    times = [0.0, 1.0, 2.0]
    pos = np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]], dtype=float)
    # 2 s at 3.6 km/h = 2 m real; colmap length 1.0 -> scale 2
    assert abs(compute_scale(times, pos, speed_kmh=3.6) - 2.0) < 1e-9


def test_prepare_colmap_masks(tmp_path):
    p = Project(tmp_path / "sec")
    ensure_dir(p.views_masks_dir)
    m = np.full((8, 8), 255, np.uint8)
    cv2.imwrite(str(p.views_masks_dir / "frame_000000_y+090.png"), m)
    out = prepare_colmap_masks(p, tmp_path / "cm")
    assert (out / "frame_000000_y+090.jpg.png").exists()


def test_build_feature_cmd_pins_intrinsics():
    intr = {"width": 1600, "height": 1600, "fx": 671.3, "fy": 671.3,
            "cx": 800.0, "cy": 800.0}
    cmd = build_feature_cmd("colmap.exe", "db.db", "imgs", "masks", intr)
    s = " ".join(str(c) for c in cmd)
    assert "--ImageReader.camera_model PINHOLE" in s
    assert "--ImageReader.single_camera 1" in s
    assert "671.3,671.3,800.0,800.0" in s
    assert "--ImageReader.mask_path masks" in s


def test_ingest_model(tmp_path):
    p = Project(tmp_path / "sec")
    intr = {"width": 128, "height": 128, "fov_deg": 90.0, "fx": 64.0, "fy": 64.0,
            "cx": 64.0, "cy": 64.0, "yaws": [-90, 90]}
    write_json(p.views_meta, intr)
    # rig moves +x at 0.1 colmap-units per frame, 10 frames, t = 0.1 s apart
    frames, model_images = [], []
    for i in range(10):
        T = _rig_pose(0.1 * i)
        frames.append({"name": f"frame_{i:06d}.jpg", "src_index": i,
                       "t": 0.1 * i, "sharpness": 1.0})
        for yaw in (-90, 90):
            model_images.append({"name": f"frame_{i:06d}_y{yaw:+04d}.jpg",
                                 "T_wc": yaw_view_pose(T, yaw)})
    write_jsonl(p.frames_meta, frames)
    model_dir = tmp_path / "model"
    pts = np.array([[0.5, 0.0, 1.0, 10, 20, 30]])
    write_colmap_model(model_dir, intr, model_images, pts)

    # colmap dist = 0.9 units over 0.9 s; at 3.6 km/h real = 0.9 m -> scale 1.0... use 7.2 -> 2.0
    ingest_model(p, model_dir, speed_kmh=7.2)

    poses = read_json(p.poses_json)
    assert abs(poses["scale"] - 2.0) < 1e-6
    assert abs(poses["frames"][-1]["s"] - 1.8) < 1e-6      # 0.9 * 2
    imgs = load_images_txt(p.colmap_dir / "images.txt")
    assert len(imgs) == 20
    p3d = (p.colmap_dir / "points3D.txt").read_text()
    assert "1.0 0.0 2.0" in p3d                             # point scaled by 2
    assert p.qc_trajectory.exists()
