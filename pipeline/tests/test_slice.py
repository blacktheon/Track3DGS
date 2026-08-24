import numpy as np
from plyfile import PlyData, PlyElement
from track3dgs.slice import gaussian_s_values, assign_tiles, prune_mask, run_slice
from track3dgs.io_utils import Project, ensure_dir, write_json


def test_gaussian_s_values_straight_line():
    traj = np.stack([np.arange(0, 101, 1.0), np.zeros(101), np.zeros(101)], axis=1)
    s = np.arange(0, 101, 1.0)
    xyz = np.array([[10.2, 0, 3.0], [55.0, 0, -7.0]])
    gs, gd = gaussian_s_values(xyz, traj, s)
    assert abs(gs[0] - 10.0) < 0.61 and abs(gd[0] - 3.0) < 0.31
    assert abs(gs[1] - 55.0) < 0.51 and abs(gd[1] - 7.0) < 0.31


def test_assign_tiles_pad_dropped():
    s = np.array([1.0, 12.0, 49.0, 55.0, -3.0])
    t = assign_tiles(s, core_lo=0.0, core_hi=50.0, tile_len=10.0)
    assert list(t) == [0, 1, 4, -1, -1]


def test_prune_mask():
    dist = np.array([5.0, 100.0, 5.0])
    op_raw = np.array([3.0, 3.0, -10.0])       # sigmoid: .95, .95, ~4.5e-5
    m = prune_mask(dist, op_raw, max_dist=60.0, min_opacity=0.005)
    assert list(m) == [True, False, False]


def _make_cell_ply(path, xyz, opacity_raw):
    names = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("f_dc_0", "f4"),
             ("opacity", "f4"), ("scale_0", "f4"), ("rot_0", "f4")]
    v = np.zeros(len(xyz), dtype=names)
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["opacity"] = opacity_raw
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))


def test_run_slice(tmp_path):
    p = Project(tmp_path / "sec")
    frames = []
    for i in range(6):                         # straight line along +x, s=0..50
        T = np.eye(4); T[:3, 3] = [i * 10.0, 0, 0]
        frames.append({"name": f"f{i}.jpg", "t": float(i), "s": float(i * 10),
                       "T_wc": [float(v) for v in T.reshape(-1)]})
    write_json(p.poses_json, {"scale": 1.0, "frames": frames})
    write_json(p.cells_json, {"cell_length": 50.0, "pad": 10.0, "tile_length": 10.0,
                              "total_s": 50.0,
                              "cells": [{"id": 0, "s_core": [0.0, 50.0],
                                         "s_full": [0.0, 50.0], "frames": []}]})
    ensure_dir(p.export_dir)
    xyz = np.array([[5.0, 0, 2.0], [15.0, 0, 2.0], [15.0, 0, 200.0]])
    _make_cell_ply(p.export_dir / "cell_000.ply", xyz, np.array([3.0, 3.0, 3.0]))
    run_slice(p.root, 0, max_dist=60.0, min_opacity=0.005)
    t0 = PlyData.read(str(p.tiles_dir / "tile_0000.ply"))["vertex"]
    t1 = PlyData.read(str(p.tiles_dir / "tile_0001.ply"))["vertex"]
    assert len(t0) == 1 and len(t1) == 1        # far gaussian pruned
    assert abs(t1["x"][0] - 15.0) < 1e-6
    assert "f_dc_0" in t1.data.dtype.names      # all props preserved
