import numpy as np
from plyfile import PlyData, PlyElement
from track3dgs.io_utils import Project, ensure_dir, write_json, read_json
from track3dgs.trajectory import yaw_view_pose
from track3dgs.pack import (estimate_alignment, transform_splats, tile_record,
                            run_pack)


def _tilted_frames(tilt_deg=30.0):
    """Rig moving +x, camera 'up' tilted about the z axis by tilt_deg."""
    a = np.radians(tilt_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0],
                   [np.sin(a), np.cos(a), 0],
                   [0, 0, 1]])
    frames = []
    for i in range(5):
        T = np.eye(4)
        T[:3, :3] = Rz
        T[:3, 3] = [2.0 * i, 0, 0]
        frames.append({"name": f"f{i}.jpg", "t": float(i), "s": 2.0 * i,
                       "T_wc": [float(v) for v in T.reshape(-1)]})
    return frames


def test_estimate_alignment_levels_up_and_faces_z():
    frames = _tilted_frames(30.0)
    R, t0 = estimate_alignment(frames)
    T0 = np.array(frames[0]["T_wc"]).reshape(4, 4)
    up_world = -T0[:3, 1]                                   # camera -Y = up
    up_aligned = R @ up_world
    assert np.allclose(up_aligned, [0, 1, 0], atol=1e-6)    # up -> +Y (viewer)
    assert np.allclose(t0, [0, 0, 0], atol=1e-9)            # start at origin
    travel = R @ np.array([1.0, 0, 0])                      # rig moved along +x
    travel[1] = 0.0                                         # ignore climb component
    travel /= np.linalg.norm(travel)
    assert travel[2] > 0.99                                 # horizontally faces +Z


def test_estimate_alignment_prefers_ground_plane():
    frames = _tilted_frames(0.0)                            # cameras claim up=-y
    rng = np.random.default_rng(1)
    # ground plane y = +2 tilted 20deg about z; normal (sin,  -cos, 0)... build
    # points on a plane whose normal is NOT the camera up axis
    a = np.radians(20.0)
    n = np.array([np.sin(a), -np.cos(a), 0.0])              # points "up" (-y-ish)
    u = np.array([np.cos(a), np.sin(a), 0.0])
    w = np.array([0.0, 0.0, 1.0])
    uv = rng.uniform(-10, 10, (2000, 2))
    pts = uv[:, :1] * u + uv[:, 1:] * w + n * (-2.0)        # 2 m below cameras
    R, _ = estimate_alignment(frames, points=pts)
    assert np.allclose(R @ n, [0, 1, 0], atol=0.02)         # ground normal -> +Y


def _splat_dtype(with_sh=True):
    names = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
             ("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"),
             ("scale_2", "f4"),
             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    if with_sh:
        names += [(f"f_rest_{i}", "f4") for i in range(45)]
    return names


def test_transform_splats_rotates_and_strips():
    v = np.zeros(1, dtype=_splat_dtype(with_sh=True))
    v["x"], v["y"], v["z"] = 1.0, 0.0, 0.0
    v["rot_0"] = 1.0                                         # identity quat wxyz
    a = np.pi / 2
    R = np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a), np.cos(a), 0], [0, 0, 1]])     # +90deg about z
    out = transform_splats(v, R, np.array([0.0, 0, 0]), strip_sh=True)
    assert "f_rest_0" not in out.dtype.names
    assert abs(out["y"][0] - 1.0) < 1e-6 and abs(out["x"][0]) < 1e-6
    q = np.array([out["rot_0"][0], out["rot_1"][0], out["rot_2"][0],
                  out["rot_3"][0]])
    assert abs(np.linalg.norm(q) - 1.0) < 1e-6
    assert abs(abs(q[3]) - np.sin(a / 2)) < 1e-6             # rotation moved into quat


def _make_project(tmp_path):
    p = Project(tmp_path / "sec")
    ensure_dir(p.tiles_dir)
    for tid, xs in [(0, [1.0, 2.0]), (1, [11.0])]:
        v = np.zeros(len(xs), dtype=_splat_dtype(with_sh=True))
        v["x"] = xs
        v["rot_0"] = 1.0
        PlyData([PlyElement.describe(v, "vertex")]).write(
            str(p.tiles_dir / f"tile_{tid:04d}.ply"))
    write_json(p.cells_json, {"cell_length": 50.0, "pad": 10.0,
                              "tile_length": 10.0, "total_s": 20.0, "cells": []})
    write_json(p.poses_json, {"scale": 1.0, "frames": _tilted_frames(0.0)})
    return p


def test_run_pack_manifest_and_merge(tmp_path):
    p = _make_project(tmp_path)
    run_pack(p.root, merge=True)
    m = read_json(p.manifest_json)
    assert m["version"] == 1 and len(m["tiles"]) == 2
    assert m["tiles"][0]["num_splats"] == 2
    assert m["tiles"][0]["s_start"] == 0.0 and m["tiles"][1]["s_start"] == 10.0
    assert len(m["alignment"]["rotation"]) == 9
    merged = PlyData.read(str(p.demo_dir / "section_merged.ply"))["vertex"]
    assert len(merged) == 3
    assert "f_rest_0" not in merged.data.dtype.names
    packed = PlyData.read(str(p.tiles_dir / "packed" / "tile_0000.ply"))["vertex"]
    assert len(packed) == 2
