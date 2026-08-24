import numpy as np
from track3dgs.cells import plan_cells, invert_wc, run_cells
from track3dgs.io_utils import Project, read_json, write_json
from track3dgs.colmap_export import write_colmap_model, load_images_txt
from track3dgs.trajectory import world_to_camera


def test_plan_cells_padding_and_remainder():
    cells = plan_cells(112.0, cell=50.0, pad=10.0)
    assert len(cells) == 2                     # 12 m remainder < 25 -> absorbed
    assert cells[0]["s_core"] == [0.0, 50.0]
    assert cells[0]["s_full"] == [0.0, 60.0]
    assert cells[1]["s_core"] == [50.0, 112.0]
    assert cells[1]["s_full"] == [40.0, 112.0]


def test_plan_cells_short_section_is_one_cell():
    cells = plan_cells(34.2, cell=50.0, pad=10.0)
    assert len(cells) == 1
    assert cells[0]["s_core"] == [0.0, 34.2]


def test_invert_wc_roundtrip():
    T = np.eye(4); T[:3, 3] = [1, 2, 3]
    qw, qx, qy, qz, tx, ty, tz = world_to_camera(T)
    T2 = invert_wc([qw, qx, qy, qz], [tx, ty, tz])
    assert np.allclose(T, T2, atol=1e-9)


def _make_project(tmp_path):
    p = Project(tmp_path / "sec")
    intr = {"width": 128, "height": 128, "fov_deg": 90.0, "fx": 64.0, "fy": 64.0,
            "cx": 64.0, "cy": 64.0, "yaws": [90]}
    write_json(p.views_meta, intr)
    frames, images = [], []
    for i in range(12):                         # rig frames every 10 m: s=0..110
        T = np.eye(4); T[:3, 3] = [10.0 * i, 0, 0]
        name = f"frame_{i:06d}.jpg"
        frames.append({"name": name, "t": float(i), "s": 10.0 * i,
                       "T_wc": [float(v) for v in T.reshape(-1)]})
        images.append({"name": f"frame_{i:06d}_y+090.jpg", "T_wc": T})
    write_json(p.poses_json, {"scale": 1.0, "frames": frames})
    pts = np.array([[5.0, 0, 2.0, 10, 20, 30], [65.0, 0, 2.0, 10, 20, 30],
                    [500.0, 0, 0, 10, 20, 30]])
    write_colmap_model(p.colmap_dir, intr, images, pts)
    return p


def test_run_cells(tmp_path):
    p = _make_project(tmp_path)
    run_cells(p.root, cell=50.0, pad=10.0, tile=10.0)
    cj = read_json(p.cells_json)
    assert len(cj["cells"]) == 2
    c0 = cj["cells"][0]
    # s_full = [0,60] -> frames s=0..60 inclusive
    assert c0["frames"] == [f"frame_{i:06d}.jpg" for i in range(7)]
    imgs = load_images_txt(p.cells_dir / "cell_000" / "colmap" / "images.txt")
    assert len(imgs) == 7                       # one yaw
    p3d = (p.cells_dir / "cell_000" / "colmap" / "points3D.txt").read_text()
    assert "5.0" in p3d                         # nearby point kept
    assert "500.0" not in p3d                   # far outlier excluded by bbox
