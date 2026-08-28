import numpy as np
from track3dgs.io_utils import Project, ensure_dir, write_json
from track3dgs.colmap_export import write_colmap_model
from track3dgs.gstrain import load_cell_dataset, project_depths
from track3dgs.trajectory import yaw_view_pose
import cv2


def _make_cell(tmp_path):
    p = Project(tmp_path / "sec")
    intr = {"width": 64, "height": 64, "fov_deg": 90.0, "fx": 32.0, "fy": 32.0,
            "cx": 32.0, "cy": 32.0, "yaws": [90]}
    write_json(p.views_meta, intr)
    ensure_dir(p.views_dir); ensure_dir(p.views_masks_dir)
    images = []
    for i in range(3):
        T = np.eye(4); T[:3, 3] = [0, 0, 2.0 * i]
        name = f"frame_{i:06d}_y+090.jpg"
        images.append({"name": name, "T_wc": T})
        cv2.imwrite(str(p.views_dir / name),
                    np.full((64, 64, 3), 100 + 20 * i, np.uint8))
        m = np.full((64, 64), 255, np.uint8); m[:8, :] = 0
        cv2.imwrite(str(p.views_masks_dir / name.replace(".jpg", ".png")), m)
    pts = np.array([[0.0, 0.0, 5.0, 200, 10, 10], [1.0, -1.0, 6.0, 10, 200, 10]])
    cdir = p.cells_dir / "cell_000" / "colmap"
    write_colmap_model(cdir, intr, images, pts)
    return p


def test_load_cell_dataset(tmp_path):
    p = _make_cell(tmp_path)
    ds = load_cell_dataset(p, 0)
    assert ds["images"].shape == (3, 64, 64, 3) and ds["images"].dtype == np.uint8
    assert ds["masks"].shape == (3, 64, 64)
    assert ds["K"].shape == (3, 3) and ds["K"][0, 0] == 32.0
    assert ds["w2c"].shape == (3, 4, 4)
    assert ds["points"].shape == (2, 3) and ds["rgbs"].shape == (2, 3)
    # w2c of first view (identity pose) is identity
    assert np.allclose(ds["w2c"][0], np.eye(4), atol=1e-9)


def test_project_depths(tmp_path):
    p = _make_cell(tmp_path)
    ds = load_cell_dataset(p, 0)
    per_view = project_depths(ds["points"], ds["w2c"], ds["K"], 64, 64)
    u, v, d = per_view[0]                       # cam at origin, pt at z=5
    assert len(d) >= 1
    assert abs(d[0] - 5.0) < 1e-6
    assert abs(u[0] - 32.0) < 1.0 and abs(v[0] - 32.0) < 1.0
    u2, v2, d2 = per_view[2]                    # cam at z=4, pt at z=5 -> d=1
    assert abs(d2[0] - 1.0) < 1e-6
