import cv2
import numpy as np
import pytest
from track3dgs.extract import sharpness, select_sharp, run_extract, probe_fps
from track3dgs.io_utils import Project, read_jsonl


def test_sharpness_orders_blur():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert sharpness(sharp) > sharpness(blurred)


def test_select_sharp_picks_group_max():
    scores = [1.0, 5.0, 2.0,  9.0, 0.5, 0.6,  3.0]
    assert select_sharp(scores, group=3) == [1, 3, 6]


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (128, 64))
    rng = np.random.default_rng(1)
    for i in range(30):
        frame = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
        if i % 3 != 0:  # 2 of every 3 frames blurred
            frame = cv2.GaussianBlur(frame, (15, 15), 5)
        w.write(frame)
    w.release()
    return path


def test_probe_fps(tiny_video):
    assert abs(probe_fps(tiny_video) - 30.0) < 0.01


def test_run_extract(tiny_video, tmp_path):
    out = tmp_path / "sec"
    run_extract(tiny_video, out, start=0.0, end=1.0, group=3, proxy_width=64)
    p = Project(out)
    meta = read_jsonl(p.frames_meta)
    assert len(meta) == 10                      # 30 frames / group 3
    assert all(m["src_index"] % 3 == 0 for m in meta)   # sharp ones win
    assert p.section_video.exists() and p.proxy_video.exists()
    assert p.mask_path.exists() and p.mask_reference.exists()
    mask = cv2.imread(str(p.mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask.min() == 255                    # all-white template
    for m in meta:
        assert (p.frames_dir / m["name"]).exists()
        assert abs(m["t"] - m["src_index"] / 30.0) < 1e-3
