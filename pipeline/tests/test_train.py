import numpy as np
from plyfile import PlyData, PlyElement
from track3dgs.io_utils import Project, ensure_dir
from track3dgs.train import build_train_cmd, build_export_cmd, check_alignment


def test_build_train_cmd_disables_normalization(tmp_path):
    p = Project(tmp_path / "sec")
    cmd = build_train_cmd(p, 0, iters=30000)
    s = " ".join(str(c) for c in cmd)
    assert "--center-method none" in s
    assert "--orientation-method none" in s
    assert "--auto-scale-poses False" in s
    assert str(p.cells_dir / "cell_000") in s
    assert "--masks-path" in s


def test_build_export_cmd(tmp_path):
    cmd = build_export_cmd(tmp_path / "config.yml", tmp_path / "out")
    assert cmd[:2] == ["ns-export", "gaussian-splat"]


def _write_ply(path, xyz):
    v = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("opacity", "f4")])
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))


def test_check_alignment(tmp_path):
    colmap = ensure_dir(tmp_path / "colmap")
    (colmap / "points3D.txt").write_text("# hdr\n1 10 0 0 128 128 128 0.5\n"
                                         "2 20 0 0 128 128 128 0.5\n")
    good = tmp_path / "good.ply"
    _write_ply(good, np.array([[14.0, 0, 0], [16.0, 0, 0]]))
    bad = tmp_path / "bad.ply"
    _write_ply(bad, np.array([[500.0, 0, 0]]))
    assert check_alignment(good, colmap)["ok"]
    assert not check_alignment(bad, colmap)["ok"]
