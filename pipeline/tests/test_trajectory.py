import numpy as np
from track3dgs.trajectory import (quat_to_R, R_to_quat, load_tum, apply_scale,
                                  yaw_view_pose, world_to_camera, arc_length,
                                  resample_polyline)


def test_quat_identity_roundtrip():
    R = quat_to_R(0, 0, 0, 1)
    assert np.allclose(R, np.eye(3))
    qw, qx, qy, qz = R_to_quat(R)
    assert abs(qw) > 0.999


def test_load_tum_and_scale(tmp_path):
    f = tmp_path / "traj.txt"
    f.write_text("# comment\n0.0 0 0 0 0 0 0 1\n1.0 1 0 0 0 0 0 1\n")
    poses = load_tum(f)
    assert len(poses) == 2 and poses[1][0] == 1.0
    scaled = apply_scale([T for _, T in poses], 2.5)
    assert np.allclose(scaled[1][:3, 3], [2.5, 0, 0])


def test_yaw_view_pose_looks_right():
    T = np.eye(4)                      # camera at origin, z-forward, x-right
    Tv = yaw_view_pose(T, 90.0)
    p_world = np.array([5.0, 0.0, 0.0, 1.0])       # point to the RIGHT
    p_cam = np.linalg.inv(Tv) @ p_world
    assert p_cam[2] > 4.9                          # ahead of the +90 view
    assert abs(p_cam[0]) < 1e-6 and abs(p_cam[1]) < 1e-6


def test_world_to_camera_recenters():
    T = np.eye(4); T[:3, 3] = [3, 0, 0]
    qw, qx, qy, qz, tx, ty, tz = world_to_camera(T)
    assert np.allclose([tx, ty, tz], [-3, 0, 0])


def test_arc_length_and_resample():
    pts = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], dtype=float)
    s = arc_length(pts)
    assert np.allclose(s, [0, 3, 7])
    dense, ds = resample_polyline(pts, step=0.5)
    assert abs(ds[-1] - 7.0) < 0.5 and len(dense) == len(ds)
    assert np.allclose(dense[0], [0, 0, 0])
