import cv2
import numpy as np
from track3dgs.automask import compute_static_mask, run_automask
from track3dgs.io_utils import Project, ensure_dir, write_jsonl


def test_compute_static_mask_flags_static_region():
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(12):
        f = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
        f[40:, :] = (0, 128, 0)          # static "vehicle" band, identical every frame
        frames.append(f)
    mask = compute_static_mask(frames, var_thresh=5.0)
    assert mask.shape == (64, 128)
    assert mask[60, 64] == 0             # static region masked out (black)
    assert mask[10, 64] == 255           # moving region kept (white)


def test_run_automask_writes_files(tmp_path):
    p = Project(tmp_path / "sec")
    ensure_dir(p.frames_dir)
    rng = np.random.default_rng(1)
    recs = []
    for i in range(6):
        f = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
        f[40:, :] = (0, 128, 0)
        name = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(p.frames_dir / name), f)
        recs.append({"name": name, "src_index": i, "t": i / 30.0, "sharpness": 1.0})
    write_jsonl(p.frames_meta, recs)
    run_automask(p.root, work_width=128)
    mask = cv2.imread(str(p.mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (64, 128)       # full frame resolution
    assert mask[60, 64] == 0 and mask[10, 64] == 255
    assert (p.root / "mask_overlay.jpg").exists()
