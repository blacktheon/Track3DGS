import math
import cv2
import numpy as np
from track3dgs.views import intrinsics, view_name, run_views
from track3dgs.io_utils import Project, write_jsonl, ensure_dir, read_json


def test_intrinsics_math():
    k = intrinsics(90.0, 1000)
    assert abs(k["fx"] - 500.0) < 1e-6          # tan(45deg)=1
    assert k["cx"] == 500.0 and k["width"] == 1000


def test_view_name():
    assert view_name("frame_000123.jpg", 90) == "frame_000123_y+090.jpg"
    assert view_name("frame_000123.jpg", -45) == "frame_000123_y-045.jpg"


def _make_project(tmp_path):
    p = Project(tmp_path / "sec")
    ensure_dir(p.frames_dir)
    # Equirect 512x256: paint a red block at longitude +90deg (x = 3/4 * width).
    eq = np.zeros((256, 512, 3), dtype=np.uint8)
    eq[96:160, 352:416] = (0, 0, 255)
    cv2.imwrite(str(p.frames_dir / "frame_000000.jpg"), eq)
    write_jsonl(p.frames_meta, [{"name": "frame_000000.jpg", "src_index": 0,
                                 "t": 0.0, "sharpness": 1.0}])
    mask = np.full((256, 512), 255, dtype=np.uint8)
    mask[160:, :] = 0    # "vehicle" band from latitude -22.5deg down (visible in 90deg FOV)
    cv2.imwrite(str(p.mask_path), mask)
    return p


def test_run_views_combines_sky_mask(tmp_path):
    from track3dgs.io_utils import ensure_dir as ed
    p = _make_project(tmp_path)
    # per-frame sky mask: top third excluded (visible in a 90deg FOV crop)
    sky = np.full((256, 512), 255, dtype=np.uint8)
    sky[:96, :] = 0
    ed(p.root / "sky_masks")
    cv2.imwrite(str(p.root / "sky_masks" / "frame_000000.png"), sky)
    run_views(p.root, yaws=[90], fov=90.0, size=128)
    m = cv2.imread(str(p.views_masks_dir / "frame_000000_y+090.png"),
                   cv2.IMREAD_GRAYSCALE)
    assert m[5, 64] == 0                        # top of crop: sky excluded
    assert m[64, 64] == 255                     # centre kept
    assert m[120, 64] == 0                      # bottom: vehicle mask still active


def test_run_views(tmp_path):
    p = _make_project(tmp_path)
    run_views(p.root, yaws=[-90, 90], fov=90.0, size=128)
    img_r = cv2.imread(str(p.views_dir / "frame_000000_y+090.jpg"))
    img_l = cv2.imread(str(p.views_dir / "frame_000000_y-090.jpg"))
    c = 64
    assert img_r[c, c, 2] > 150 and img_r[c, c, 0] < 80   # red at center of +90 view
    assert img_l[c, c, 2] < 80                            # not in -90 view
    m = cv2.imread(str(p.views_masks_dir / "frame_000000_y+090.png"),
                   cv2.IMREAD_GRAYSCALE)
    assert m.shape == (128, 128) and m.max() == 255 and m.min() == 0
    meta = read_json(p.views_meta)
    assert meta["yaws"] == [-90, 90] and meta["width"] == 128
