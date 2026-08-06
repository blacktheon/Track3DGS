# Track3DGS Phase 1 — Offline Reconstruction Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the seven-stage offline pipeline (`extract → views → track → cells → train → slice → pack`), prove it end-to-end on a 30–40 m test section of the 360° track video, and produce a desktop investor demo (flythrough video + interactive viewer).

**Architecture:** A Python package `track3dgs` of seven standalone CLI stages sharing an on-disk project layout. Global camera trajectory comes from stella_vslam (Docker) on the equirectangular video; training uses Nerfstudio Splatfacto/gsplat per ~50 m cell **with pose normalization disabled** so all outputs live in one fixed global metric frame; trained cells are sliced into non-overlapping 10 m tiles by arc-length. Spec: `docs/superpowers/specs/2026-08-06-track3dgs-pipeline-design.md`.

**Tech Stack:** Python 3.11, FFmpeg, OpenCV, py360convert, NumPy/SciPy, plyfile, msgpack, stella_vslam (Docker), Nerfstudio ≥1.1 + gsplat (CUDA), matplotlib (QC plots), pytest.

## Global Constraints

- Runtime tile length: **10 m**; training cell: **50 m core + 10 m padding each side** (spec §2, §4).
- All poses, point clouds, and PLYs stay in **one fixed global metric frame** — never re-center, re-orient, or re-scale during training (spec §2).
- Provisional training cap: **2–3 M Gaussians per cell**; target ≈ **400 K per tile** after pruning (spec §3 Phase 1).
- Every Gaussian belongs to **exactly one tile**; tiles never overlap (spec §2).
- Source footage: **7680×3840, 30 fps, mp4** (spec §1).
- No manual touch-ups in any stage — everything scripted and re-runnable per cell (spec §3 Phase 1).
- Viewer tube: reconstruction only needs to look good within ~1–2 m of the capture trajectory (spec §1).
- Vehicle-body mask must be applied to all training imagery (spec §4 stage 1).
- Perspective views: 4–6 yaw directions, ~100° FOV, ~1600 px, one shared pinhole intrinsic (spec §4 stage 2).
- Training host: Windows workstation w/ RTX 5060 Ti (16 GB, Blackwell sm_120 — needs CUDA 12.8-class torch); DGX Spark optional later.
- Camera/axis convention everywhere: **OpenCV/COLMAP** (x-right, y-down, z-forward); TUM pose files are camera-to-world; COLMAP `images.txt` stores world-to-camera.

## File Structure

```
pipeline/
  requirements.txt
  pytest.ini
  track3dgs/
    __init__.py
    io_utils.py        # project layout + json/jsonl helpers
    extract.py         # stage 1 CLI: section trim, frames, sharpness, mask template
    views.py           # stage 2 CLI: equirect → perspective crops + masks + intrinsics
    trajectory.py      # pose math: TUM, quats, yaw offsets, scale, arc-length
    colmap_export.py   # write COLMAP text model
    track.py           # stage 3 CLI: ingest VSLAM output → global poses + COLMAP model + QC plot
    cells.py           # stage 4 CLI: arc-length cells + per-cell COLMAP subsets
    train.py           # stage 5 CLI: ns-train/ns-export wrapper + alignment check
    slice.py           # stage 6 CLI: cell PLY → 10 m tile PLYs (+ pruning)
    pack.py            # stage 7 CLI: tile manifest + merged section PLY
    demo_camera_path.py# investor demo: trajectory → nerfstudio camera-path JSON
  tests/
    test_io_utils.py
    test_extract.py
    test_views.py
    test_trajectory.py
    test_colmap_export.py
    test_track.py
    test_cells.py
    test_train.py
    test_slice.py
    test_pack.py
    test_demo_camera_path.py
data/                  # gitignored: raw video + per-section outputs
```

Per-section data layout (created by the stages, all paths via `io_utils.Project`):

```
data/section01/
  section.mp4            # trimmed 30fps CFR section video (t=0 at section start)
  proxy_vslam.mp4        # 1920x960 proxy for stella_vslam
  frames/frame_%06d.jpg  # selected sharp frames (8K equirect)
  frames_meta.jsonl      # {name, src_index, t, sharpness}
  mask_equirect.png      # white=keep, black=vehicle; hand-painted once from mask_reference.jpg
  mask_reference.jpg
  views/                 # {frame}_y{+yaw}.jpg perspective crops
  views_masks/           # same filenames, per-view mask PNGs
  views_meta.json        # {width,height,fov_deg,fx,fy,cx,cy,yaws}
  track/frame_trajectory.txt  # TUM from stella_vslam (mounted output)
  track/map.msg               # stella_vslam map dump
  track/poses.json            # per rig-frame global T_wc + arc-length s
  track/colmap/{cameras,images,points3D}.txt   # full-section model (per-view poses)
  track/qc_trajectory.png
  cells/cells.json
  cells/cell_000/colmap/{cameras,images,points3D}.txt
  train/cell_000/…            # nerfstudio run output
  train/export/cell_000.ply   # global-frame gaussians
  tiles/tile_0000.ply …
  tiles/manifest.json
  demo/camera_path.json, demo/flythrough.mp4, demo/section_merged.ply
```

---

### Task 1: Scaffold + `io_utils`

**Files:**
- Create: `pipeline/requirements.txt`, `pipeline/pytest.ini`, `pipeline/track3dgs/__init__.py`, `pipeline/track3dgs/io_utils.py`, `pipeline/tests/test_io_utils.py`, `.gitignore`, `README.md`

**Interfaces:**
- Produces: `Project(root)` with properties `frames_dir, frames_meta, mask_path, mask_reference, section_video, proxy_video, views_dir, views_masks_dir, views_meta, track_dir, poses_json, colmap_dir, qc_trajectory, cells_dir, cells_json, train_dir, export_dir, tiles_dir, manifest_json, demo_dir` (all `pathlib.Path`, all under `root`); `read_json/write_json(path, obj)`, `read_jsonl/write_jsonl(path, records)`; `ensure_dir(path) -> Path`.

- [ ] **Step 1: Create environment and scaffold**

```powershell
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
conda create -n track3dgs python=3.11 -y
conda activate track3dgs
```

`pipeline/requirements.txt`:
```
numpy>=1.26
opencv-python>=4.9
py360convert>=1.0
plyfile>=1.0
scipy>=1.11
matplotlib>=3.8
msgpack>=1.0
tqdm>=4.66
pytest>=8.0
```

```powershell
pip install -r pipeline/requirements.txt
```

`pipeline/pytest.ini`:
```ini
[pytest]
testpaths = tests
```

`.gitignore` (repo root):
```
data/
__pycache__/
*.pyc
.pytest_cache/
pipeline/train/
*.ply
*.msg
```

`README.md` (repo root): title, one-paragraph description, pointer to spec and plan docs, `conda activate track3dgs` + `pip install -r pipeline/requirements.txt` setup lines, and the seven-stage command sequence from Task 12 Step 1 (copy it there once written).

- [ ] **Step 2: Write failing tests**

`pipeline/tests/test_io_utils.py`:
```python
import json
from pathlib import Path
from track3dgs.io_utils import Project, read_json, write_json, read_jsonl, write_jsonl, ensure_dir

def test_project_paths(tmp_path):
    p = Project(tmp_path / "section01")
    assert p.frames_dir == tmp_path / "section01" / "frames"
    assert p.views_meta == tmp_path / "section01" / "views" / "views_meta.json"
    assert p.manifest_json == tmp_path / "section01" / "tiles" / "manifest.json"
    assert p.colmap_dir == tmp_path / "section01" / "track" / "colmap"

def test_json_roundtrip(tmp_path):
    f = tmp_path / "x.json"
    write_json(f, {"a": 1})
    assert read_json(f) == {"a": 1}

def test_jsonl_roundtrip(tmp_path):
    f = tmp_path / "x.jsonl"
    write_jsonl(f, [{"i": 0}, {"i": 1}])
    assert read_jsonl(f) == [{"i": 0}, {"i": 1}]

def test_ensure_dir(tmp_path):
    d = ensure_dir(tmp_path / "a" / "b")
    assert d.is_dir()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_io_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'track3dgs'` (or ImportError).

- [ ] **Step 4: Implement `io_utils.py`**

`pipeline/track3dgs/io_utils.py`:
```python
import json
from pathlib import Path


class Project:
    """On-disk layout for one track section. All stages read/write through this."""

    def __init__(self, root):
        self.root = Path(root)

    @property
    def section_video(self): return self.root / "section.mp4"
    @property
    def proxy_video(self): return self.root / "proxy_vslam.mp4"
    @property
    def frames_dir(self): return self.root / "frames"
    @property
    def frames_meta(self): return self.root / "frames_meta.jsonl"
    @property
    def mask_path(self): return self.root / "mask_equirect.png"
    @property
    def mask_reference(self): return self.root / "mask_reference.jpg"
    @property
    def views_dir(self): return self.root / "views"
    @property
    def views_masks_dir(self): return self.root / "views_masks"
    @property
    def views_meta(self): return self.root / "views" / "views_meta.json"
    @property
    def track_dir(self): return self.root / "track"
    @property
    def poses_json(self): return self.root / "track" / "poses.json"
    @property
    def colmap_dir(self): return self.root / "track" / "colmap"
    @property
    def qc_trajectory(self): return self.root / "track" / "qc_trajectory.png"
    @property
    def cells_dir(self): return self.root / "cells"
    @property
    def cells_json(self): return self.root / "cells" / "cells.json"
    @property
    def train_dir(self): return self.root / "train"
    @property
    def export_dir(self): return self.root / "train" / "export"
    @property
    def tiles_dir(self): return self.root / "tiles"
    @property
    def manifest_json(self): return self.root / "tiles" / "manifest.json"
    @property
    def demo_dir(self): return self.root / "demo"


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, obj):
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, records):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
```

`pipeline/track3dgs/__init__.py`: empty file.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_io_utils.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```powershell
git add .gitignore README.md pipeline/
git commit -m "feat: scaffold track3dgs pipeline package with project layout"
```

---

### Task 2: Stage 1 — `extract`

**Files:**
- Create: `pipeline/track3dgs/extract.py`, `pipeline/tests/test_extract.py`

**Interfaces:**
- Consumes: `Project`, `write_jsonl`, `ensure_dir` from `io_utils`.
- Produces: CLI `python -m track3dgs.extract --video V --out DIR --start S --end E [--group 3] [--proxy-width 1920]`. Functions: `sharpness(img_bgr) -> float` (variance of Laplacian on 960-px-wide grayscale); `select_sharp(scores: list[float], group: int) -> list[int]` (indices of per-group argmax); `run_extract(video, out, start, end, group, proxy_width)`.
- Output files: `section.mp4` (re-encoded CFR 30 fps from `--start` to `--end`), `proxy_vslam.mp4`, `frames/frame_%06d.jpg` (selected only, numbered by source index), `frames_meta.jsonl` records `{"name","src_index","t","sharpness"}` where `t = src_index / 30.0` (seconds from section start), `mask_reference.jpg` (first frame), `mask_equirect.png` (all-white template, same size).

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_extract.py`:
```python
import cv2
import numpy as np
import pytest
from track3dgs.extract import sharpness, select_sharp, run_extract
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
        assert abs(m["t"] - m["src_index"] / 30.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'track3dgs.extract'`.

- [ ] **Step 3: Implement `extract.py`**

`pipeline/track3dgs/extract.py`:
```python
"""Stage 1: trim section, extract sharp frames, build proxy + mask template."""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .io_utils import Project, ensure_dir, write_jsonl

FPS = 30.0


def sharpness(img_bgr):
    h, w = img_bgr.shape[:2]
    if w > 960:
        img_bgr = cv2.resize(img_bgr, (960, int(h * 960 / w)))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharp(scores, group):
    keep = []
    for g0 in range(0, len(scores), group):
        chunk = scores[g0:g0 + group]
        keep.append(g0 + int(np.argmax(chunk)))
    return keep


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                   check=True)


def run_extract(video, out, start, end, group=3, proxy_width=1920):
    p = Project(out)
    ensure_dir(p.root)
    # Re-encode (not stream-copy) so t=0 lands exactly on --start.
    _ffmpeg("-ss", str(start), "-to", str(end), "-i", str(video),
            "-r", str(int(FPS)), "-c:v", "libx264", "-crf", "18", "-an",
            str(p.section_video))
    _ffmpeg("-i", str(p.section_video),
            "-vf", f"scale={proxy_width}:{proxy_width // 2}",
            "-c:v", "libx264", "-crf", "23", "-an", str(p.proxy_video))

    with tempfile.TemporaryDirectory() as td:
        _ffmpeg("-i", str(p.section_video), "-qscale:v", "2", "-start_number", "0",
                str(Path(td) / "f_%06d.jpg"))
        all_frames = sorted(Path(td).glob("f_*.jpg"))
        scores = [sharpness(cv2.imread(str(f))) for f in all_frames]
        keep = select_sharp(scores, group)

        ensure_dir(p.frames_dir)
        records = []
        for idx in keep:
            name = f"frame_{idx:06d}.jpg"
            shutil.copy2(all_frames[idx], p.frames_dir / name)
            records.append({"name": name, "src_index": idx,
                            "t": idx / FPS, "sharpness": scores[idx]})
        write_jsonl(p.frames_meta, records)

        first = cv2.imread(str(all_frames[0]))
        cv2.imwrite(str(p.mask_reference), first)
        if not p.mask_path.exists():  # never clobber a hand-painted mask
            cv2.imwrite(str(p.mask_path),
                        np.full(first.shape[:2], 255, dtype=np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--proxy-width", type=int, default=1920)
    a = ap.parse_args()
    run_extract(a.video, a.out, a.start, a.end, a.group, a.proxy_width)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_extract.py -v`
Expected: 3 passed. (Requires `ffmpeg` on PATH — open a new shell after winget install.)

- [ ] **Step 5: Commit**

```powershell
git add pipeline/track3dgs/extract.py pipeline/tests/test_extract.py
git commit -m "feat: extract stage - section trim, sharp-frame selection, mask template"
```

---

### Task 3: Stage 2 — `views`

**Files:**
- Create: `pipeline/track3dgs/views.py`, `pipeline/tests/test_views.py`

**Interfaces:**
- Consumes: `Project`, `read_jsonl`, `write_json`, `ensure_dir`.
- Produces: CLI `python -m track3dgs.views --project DIR [--yaws -135,-90,-45,45,90,135] [--fov 100] [--size 1600]`. Functions: `intrinsics(fov_deg, size) -> dict` with keys `width,height,fov_deg,fx,fy,cx,cy` where `fx = fy = (size/2)/tan(radians(fov_deg)/2)`, `cx = cy = size/2`; `view_name(frame_name, yaw) -> str` returning e.g. `frame_000123_y+090.jpg`; `run_views(project_dir, yaws, fov, size)`.
- Output files: `views/{view_name}.jpg` for every frame × yaw; `views_masks/{view_name}.png` (crop of `mask_equirect.png` at same yaw — identical for all frames of a yaw but written per view name because trainers expect one mask per image); `views/views_meta.json` = `intrinsics(...) | {"yaws": [...]}`.
- Yaw convention: `u_deg=yaw` passed straight to `py360convert.e2p`; positive yaw looks toward positive longitude (right of vehicle). Task 4's `yaw_view_pose` must match this (verified end-to-end in Task 6 QC).

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_views.py`:
```python
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
    mask[200:, :] = 0                            # "vehicle" at nadir band
    cv2.imwrite(str(p.mask_path), mask)
    return p


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_views.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `views.py`**

`pipeline/track3dgs/views.py`:
```python
"""Stage 2: equirect frames -> perspective crops (+ per-view masks + intrinsics)."""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import py360convert
from tqdm import tqdm

from .io_utils import Project, ensure_dir, read_jsonl, write_json

DEFAULT_YAWS = [-135, -90, -45, 45, 90, 135]


def intrinsics(fov_deg, size):
    f = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return {"width": size, "height": size, "fov_deg": fov_deg,
            "fx": f, "fy": f, "cx": size / 2.0, "cy": size / 2.0}


def view_name(frame_name, yaw):
    stem = Path(frame_name).stem
    return f"{stem}_y{yaw:+04d}.jpg"


def _e2p(img, yaw, fov, size):
    return py360convert.e2p(img, fov_deg=(fov, fov), u_deg=yaw, v_deg=0,
                            out_hw=(size, size))


def run_views(project_dir, yaws=DEFAULT_YAWS, fov=100.0, size=1600):
    p = Project(project_dir)
    ensure_dir(p.views_dir)
    ensure_dir(p.views_masks_dir)
    frames = read_jsonl(p.frames_meta)

    mask = cv2.imread(str(p.mask_path), cv2.IMREAD_GRAYSCALE)
    mask_crops = {}
    for yaw in yaws:
        mc = _e2p(np.stack([mask] * 3, axis=-1), yaw, fov, size)[:, :, 0]
        mask_crops[yaw] = (mc > 127).astype(np.uint8) * 255

    for rec in tqdm(frames, desc="views"):
        img = cv2.imread(str(p.frames_dir / rec["name"]))
        for yaw in yaws:
            name = view_name(rec["name"], yaw)
            cv2.imwrite(str(p.views_dir / name), _e2p(img, yaw, fov, size),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(p.views_masks_dir / Path(name).with_suffix(".png").name),
                        mask_crops[yaw])

    write_json(p.views_meta, intrinsics(fov, size) | {"yaws": list(yaws)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--yaws", default=",".join(str(y) for y in DEFAULT_YAWS))
    ap.add_argument("--fov", type=float, default=100.0)
    ap.add_argument("--size", type=int, default=1600)
    a = ap.parse_args()
    run_views(a.project, [int(y) for y in a.yaws.split(",")], a.fov, a.size)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_views.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add pipeline/track3dgs/views.py pipeline/tests/test_views.py
git commit -m "feat: views stage - equirect to perspective crops with masks"
```

---

### Task 4: `trajectory` pose-math module

**Files:**
- Create: `pipeline/track3dgs/trajectory.py`, `pipeline/tests/test_trajectory.py`

**Interfaces:**
- Consumes: nothing project-specific (pure math, numpy only).
- Produces:
  - `quat_to_R(qx, qy, qz, qw) -> np.ndarray (3,3)`
  - `R_to_quat(R) -> (qw, qx, qy, qz)` (COLMAP order)
  - `load_tum(path) -> list[tuple[float, np.ndarray]]` — `(timestamp, T_wc 4x4)` per line `t tx ty tz qx qy qz qw`
  - `apply_scale(T_list, s) -> list` — multiplies translations by `s`
  - `yaw_view_pose(T_wc, yaw_deg) -> np.ndarray (4,4)` — `T_wc @ Ry(yaw)`, rotation about camera local +Y (down) axis: `Ry = [[c,0,s],[0,1,0],[-s,0,c]]`; positive yaw turns the optical axis toward camera-local +X (right), matching `views.py` positive-`u_deg` crops
  - `world_to_camera(T_wc) -> (qw, qx, qy, qz, tx, ty, tz)` — COLMAP `images.txt` fields: `R_cw = R_wc.T`, `t_cw = -R_cw @ C`
  - `arc_length(positions (N,3)) -> np.ndarray (N,)` cumulative, starting at 0
  - `resample_polyline(positions, step) -> (pts (M,3), s (M,))` — dense equal-`step` resampling for nearest-point lookup

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_trajectory.py`:
```python
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
    assert np.allclose(s, [0, 3, 8])
    dense, ds = resample_polyline(pts, step=0.5)
    assert abs(ds[-1] - 8.0) < 0.5 and len(dense) == len(ds)
    assert np.allclose(dense[0], [0, 0, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_trajectory.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `trajectory.py`**

`pipeline/track3dgs/trajectory.py`:
```python
"""Pose math. Convention: OpenCV/COLMAP camera (x-right, y-down, z-forward).
TUM lines are camera-to-world; COLMAP images.txt is world-to-camera."""
import math
from pathlib import Path

import numpy as np


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def R_to_quat(R):
    qw = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if qw > 1e-8:
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    else:  # qw ~ 0: fall back to largest diagonal element branch
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        q = np.zeros(4)
        q[i + 1] = math.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) / 2
        q[0] = (R[k, j] - R[j, k]) / (4 * q[i + 1])
        q[j + 1] = (R[j, i] + R[i, j]) / (4 * q[i + 1])
        q[k + 1] = (R[k, i] + R[i, k]) / (4 * q[i + 1])
        qw, qx, qy, qz = q
    return qw, qx, qy, qz


def load_tum(path):
    poses = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t, tx, ty, tz, qx, qy, qz, qw = (float(v) for v in line.split())
        T = np.eye(4)
        T[:3, :3] = quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = [tx, ty, tz]
        poses.append((t, T))
    return poses


def apply_scale(T_list, s):
    out = []
    for T in T_list:
        T2 = T.copy()
        T2[:3, 3] *= s
        out.append(T2)
    return out


def yaw_view_pose(T_wc, yaw_deg):
    a = math.radians(yaw_deg)
    Ry = np.array([[math.cos(a), 0, math.sin(a)],
                   [0, 1, 0],
                   [-math.sin(a), 0, math.cos(a)]])
    T = T_wc.copy()
    T[:3, :3] = T_wc[:3, :3] @ Ry
    return T


def world_to_camera(T_wc):
    R_cw = T_wc[:3, :3].T
    t = -R_cw @ T_wc[:3, 3]
    qw, qx, qy, qz = R_to_quat(R_cw)
    return qw, qx, qy, qz, t[0], t[1], t[2]


def arc_length(positions):
    d = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_polyline(positions, step):
    s = arc_length(positions)
    total = s[-1]
    s_new = np.arange(0.0, total + step, step)
    s_new = s_new[s_new <= total]
    pts = np.stack([np.interp(s_new, s, positions[:, i]) for i in range(3)], axis=1)
    return pts, s_new
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_trajectory.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```powershell
git add pipeline/track3dgs/trajectory.py pipeline/tests/test_trajectory.py
git commit -m "feat: trajectory pose math (TUM, quats, yaw views, arc-length)"
```

---

### Task 5: `colmap_export` module

**Files:**
- Create: `pipeline/track3dgs/colmap_export.py`, `pipeline/tests/test_colmap_export.py`

**Interfaces:**
- Consumes: `world_to_camera` from `trajectory`.
- Produces: `write_colmap_model(out_dir, intr: dict, images: list[dict], points: np.ndarray | None)` writing COLMAP **text** model:
  - `cameras.txt`: single line `1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}` (from `views_meta`-style dict).
  - `images.txt`: per image `IMAGE_ID QW QX QY QZ TX TY TZ 1 NAME` + blank observations line. Each `images` element: `{"name": str, "T_wc": np.ndarray(4,4)}`; IDs assigned 1..N in list order.
  - `points3D.txt`: per row of `points` `(N,3)`: `PID X Y Z 128 128 128 0.5` + no track (COLMAP tolerates empty tracks for import; used only to seed Gaussians). `points=None` writes header-only file.
- Also `load_images_txt(path) -> list[dict]` (name → quat/trans as floats) used by tests and `cells` subsetting.

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_colmap_export.py`:
```python
import numpy as np
from track3dgs.colmap_export import write_colmap_model, load_images_txt


def test_write_and_load_model(tmp_path):
    intr = {"width": 1600, "height": 1600, "fx": 671.0, "fy": 671.0,
            "cx": 800.0, "cy": 800.0}
    T = np.eye(4); T[:3, 3] = [1.0, 2.0, 3.0]
    images = [{"name": "a.jpg", "T_wc": T}]
    pts = np.array([[0.0, 0.0, 5.0]])
    write_colmap_model(tmp_path, intr, images, pts)

    cams = (tmp_path / "cameras.txt").read_text()
    assert "PINHOLE 1600 1600" in cams
    loaded = load_images_txt(tmp_path / "images.txt")
    assert loaded[0]["name"] == "a.jpg"
    assert np.allclose(loaded[0]["t"], [-1.0, -2.0, -3.0])   # world-to-camera
    p3d = (tmp_path / "points3D.txt").read_text()
    assert "0.0 0.0 5.0" in p3d or "0 0 5" in p3d.replace(".0", "")


def test_points_none(tmp_path):
    intr = {"width": 10, "height": 10, "fx": 5, "fy": 5, "cx": 5, "cy": 5}
    write_colmap_model(tmp_path, intr, [], None)
    assert (tmp_path / "points3D.txt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_colmap_export.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `colmap_export.py`**

`pipeline/track3dgs/colmap_export.py`:
```python
"""Write a COLMAP text model (cameras/images/points3D) that Nerfstudio can ingest."""
from pathlib import Path

import numpy as np

from .io_utils import ensure_dir
from .trajectory import world_to_camera


def write_colmap_model(out_dir, intr, images, points):
    out = ensure_dir(out_dir)

    (out / "cameras.txt").write_text(
        "# Camera list\n"
        f"1 PINHOLE {intr['width']} {intr['height']} "
        f"{intr['fx']} {intr['fy']} {intr['cx']} {intr['cy']}\n")

    lines = ["# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME"]
    for i, im in enumerate(images, start=1):
        qw, qx, qy, qz, tx, ty, tz = world_to_camera(im["T_wc"])
        lines.append(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {im['name']}")
        lines.append("")  # empty observations line
    (out / "images.txt").write_text("\n".join(lines) + "\n")

    plines = ["# 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]"]
    if points is not None:
        for pid, (x, y, z) in enumerate(np.asarray(points), start=1):
            plines.append(f"{pid} {x} {y} {z} 128 128 128 0.5")
    (out / "points3D.txt").write_text("\n".join(plines) + "\n")


def load_images_txt(path):
    out = []
    lines = [l for l in Path(path).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    for l in lines[::2]:  # every other line is the (empty) observations line
        f = l.split()
        out.append({"id": int(f[0]),
                    "q": [float(v) for v in f[1:5]],
                    "t": [float(v) for v in f[5:8]],
                    "name": f[9]})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_colmap_export.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add pipeline/track3dgs/colmap_export.py pipeline/tests/test_colmap_export.py
git commit -m "feat: COLMAP text model writer/reader"
```

---

### Task 6: Stage 3 — `track` (VSLAM ingest → global poses + COLMAP model)

**Files:**
- Create: `pipeline/track3dgs/track.py`, `pipeline/tests/test_track.py`, `docs/runbooks/stella-vslam.md`

**Interfaces:**
- Consumes: `Project`, `read_jsonl`, `write_json`; `trajectory.*`; `colmap_export.write_colmap_model`; `views_meta` (intrinsics + yaws).
- Produces: CLI `python -m track3dgs.track --project DIR (--speed-kmh X | --scale-known t0,t1,meters)`. Functions:
  - `match_frames_to_poses(frames_meta, tum_poses, tol=0.02) -> list[dict]` — for each extracted frame, nearest-timestamp TUM pose within `tol` s; returns `{"name","t","T_wc"}`; frames with no pose are dropped (VSLAM lost tracking there) and counted.
  - `compute_scale_from_speed(tum_poses, speed_kmh) -> float` — `real_dist / vslam_dist` where `real_dist = speed_mps * (t_last - t_first)`.
  - `load_map_points(map_msg_path) -> np.ndarray (N,3) | None` — msgpack-parse stella_vslam map dump, collect `landmarks[*]["pos_w"]`; returns `None` (with warning) on any parse failure.
  - `run_track(project_dir, speed_kmh=None, scale_known=None)`.
- Output files: `track/poses.json` = `{"scale": s, "frames": [{"name","t","s","T_wc": 16 floats row-major}]}` (rig-frame poses, scaled, arc-length `s` per frame); `track/colmap/` full-section model — one image entry **per view** (`view_name(frame, yaw)` with `yaw_view_pose`), points3D from scaled map landmarks; `track/qc_trajectory.png` (top-down XZ plot + height profile, matplotlib).

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_track.py`:
```python
import numpy as np
from track3dgs.io_utils import Project, ensure_dir, write_jsonl, write_json, read_json
from track3dgs.track import (match_frames_to_poses, compute_scale_from_speed,
                             run_track)
from track3dgs.colmap_export import load_images_txt


def _tum_line(t, x):
    return f"{t} {x} 0 0 0 0 0 1"


def _make_project(tmp_path):
    p = Project(tmp_path / "sec")
    ensure_dir(p.track_dir)
    # straight line along +x, 1 unit/s in VSLAM units, 3 s
    lines = [_tum_line(i / 10.0, i / 10.0) for i in range(31)]
    (p.track_dir / "frame_trajectory.txt").write_text("\n".join(lines))
    write_jsonl(p.frames_meta, [
        {"name": f"frame_{i:06d}.jpg", "src_index": i, "t": i / 10.0,
         "sharpness": 1.0} for i in range(31)])
    write_json(p.views_meta, {"width": 128, "height": 128, "fov_deg": 90.0,
                              "fx": 64.0, "fy": 64.0, "cx": 64.0, "cy": 64.0,
                              "yaws": [-90, 90]})
    return p


def test_scale_from_speed(tmp_path):
    p = _make_project(tmp_path)
    from track3dgs.trajectory import load_tum
    poses = load_tum(p.track_dir / "frame_trajectory.txt")
    # want 6 m/s -> 3 s * 6 = 18 m real; vslam dist = 3 units -> scale 6
    s = compute_scale_from_speed(poses, speed_kmh=21.6)
    assert abs(s - 6.0) < 1e-6


def test_match_drops_unmatched(tmp_path):
    p = _make_project(tmp_path)
    from track3dgs.trajectory import load_tum
    poses = load_tum(p.track_dir / "frame_trajectory.txt")
    frames = [{"name": "a.jpg", "t": 0.1}, {"name": "b.jpg", "t": 99.0}]
    matched = match_frames_to_poses(frames, poses)
    assert len(matched) == 1 and matched[0]["name"] == "a.jpg"


def test_run_track_outputs(tmp_path):
    p = _make_project(tmp_path)
    run_track(p.root, speed_kmh=21.6)
    poses = read_json(p.poses_json)
    assert abs(poses["scale"] - 6.0) < 1e-6
    assert abs(poses["frames"][-1]["s"] - 18.0) < 1e-3   # scaled arc-length
    imgs = load_images_txt(p.colmap_dir / "images.txt")
    assert len(imgs) == 31 * 2                            # frames x yaws
    assert imgs[0]["name"].endswith("_y-090.jpg")
    assert p.qc_trajectory.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_track.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `track.py`**

`pipeline/track3dgs/track.py`:
```python
"""Stage 3: ingest stella_vslam trajectory + map -> scaled global poses,
per-view COLMAP model, QC plot."""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msgpack
import numpy as np

from .colmap_export import write_colmap_model
from .io_utils import Project, ensure_dir, read_json, read_jsonl, write_json
from .trajectory import apply_scale, arc_length, load_tum, yaw_view_pose
from .views import view_name


def compute_scale_from_speed(tum_poses, speed_kmh):
    t0, T0 = tum_poses[0]
    t1, T1 = tum_poses[-1]
    pos = np.array([T[:3, 3] for _, T in tum_poses])
    vslam_dist = arc_length(pos)[-1]
    real_dist = (speed_kmh / 3.6) * (t1 - t0)
    return real_dist / vslam_dist


def compute_scale_known(tum_poses, t0, t1, meters):
    def pos_at(tq):
        i = int(np.argmin([abs(t - tq) for t, _ in tum_poses]))
        return tum_poses[i][1][:3, 3]
    vslam_dist = float(np.linalg.norm(pos_at(t1) - pos_at(t0)))
    return meters / vslam_dist


def match_frames_to_poses(frames_meta, tum_poses, tol=0.02):
    times = np.array([t for t, _ in tum_poses])
    out, dropped = [], 0
    for f in frames_meta:
        i = int(np.argmin(np.abs(times - f["t"])))
        if abs(times[i] - f["t"]) <= tol:
            out.append({"name": f["name"], "t": f["t"], "T_wc": tum_poses[i][1]})
        else:
            dropped += 1
    if dropped:
        warnings.warn(f"{dropped} frames had no VSLAM pose within {tol}s (tracking gaps)")
    return out


def load_map_points(map_msg_path):
    try:
        with open(map_msg_path, "rb") as f:
            data = msgpack.unpack(f, raw=False, strict_map_key=False)
        pts = [lm["pos_w"] for lm in data["landmarks"].values() if "pos_w" in lm]
        return np.array(pts, dtype=float) if pts else None
    except Exception as e:  # missing file, format drift -> random init downstream
        warnings.warn(f"could not parse map landmarks ({e}); points3D will be empty")
        return None


def _qc_plot(positions, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(positions[:, 0], positions[:, 2], ".-", ms=2)
    ax1.set_title("top-down (x,z) [m]"); ax1.set_aspect("equal")
    s = arc_length(positions)
    ax2.plot(s, -positions[:, 1])  # y-down convention -> plot height as -y
    ax2.set_title("height vs arc-length [m]")
    fig.savefig(out_path, dpi=120); plt.close(fig)


def run_track(project_dir, speed_kmh=None, scale_known=None):
    p = Project(project_dir)
    tum = load_tum(p.track_dir / "frame_trajectory.txt")
    if speed_kmh is not None:
        scale = compute_scale_from_speed(tum, speed_kmh)
    else:
        t0, t1, meters = scale_known
        scale = compute_scale_known(tum, t0, t1, meters)

    frames = match_frames_to_poses(read_jsonl(p.frames_meta), tum)
    T_scaled = apply_scale([f["T_wc"] for f in frames], scale)
    positions = np.array([T[:3, 3] for T in T_scaled])
    s_vals = arc_length(positions)

    meta = read_json(p.views_meta)
    images = [{"name": view_name(f["name"], yaw), "T_wc": yaw_view_pose(T, yaw)}
              for f, T in zip(frames, T_scaled) for yaw in meta["yaws"]]
    points = load_map_points(p.track_dir / "map.msg")
    if points is not None:
        points = points * scale
    write_colmap_model(p.colmap_dir, meta, images, points)

    write_json(p.poses_json, {"scale": scale, "frames": [
        {"name": f["name"], "t": f["t"], "s": float(sv),
         "T_wc": [float(v) for v in T.reshape(-1)]}
        for f, T, sv in zip(frames, T_scaled, s_vals)]})
    ensure_dir(p.track_dir)
    _qc_plot(positions, p.qc_trajectory)
    print(f"scale={scale:.4f}  frames={len(frames)}  views={len(images)}  "
          f"length={s_vals[-1]:.1f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--speed-kmh", type=float)
    g.add_argument("--scale-known", help="t0,t1,meters")
    a = ap.parse_args()
    sk = tuple(float(v) for v in a.scale_known.split(",")) if a.scale_known else None
    run_track(a.project, a.speed_kmh, sk)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_track.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the stella_vslam runbook**

`docs/runbooks/stella-vslam.md`:
```markdown
# Running stella_vslam on a section (Docker Desktop, WSL2 backend)

One-time build:
    git clone --recursive https://github.com/stella-cv/stella_vslam.git C:\Work\tools\stella_vslam
    cd C:\Work\tools\stella_vslam
    docker build -t stella_vslam-socket -f Dockerfile.socket .
    # vocabulary
    curl -L -o orb_vocab.fbow https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow

Config `equirect.yaml` (place next to orb_vocab.fbow):
    Camera:
      name: "insta 360"
      setup: "monocular"
      model: "equirectangular"
      fps: 30.0
      cols: 1920
      rows: 960
      color_order: "RGB"
    Feature:
      max_num_keypoints: 2000
      ini_max_num_keypoints: 4000
      scale_factor: 1.2
      num_levels: 8
      ini_fast_threshold: 20
      min_fast_threshold: 7

Run on a section (mount data dir; adjust drive path style for Docker Desktop):
    docker run --rm -v C:\Work\Unity\DSTA\Track3DGS\data:/data -v C:\Work\tools\stella_vslam:/cfg `
      stella_vslam-socket `
      /stella_vslam/build/run_video_slam `
      -v /cfg/orb_vocab.fbow -c /cfg/equirect.yaml `
      -m /data/section01/proxy_vslam.mp4 --no-sleep --auto-term `
      --eval-log-dir /data/section01/track `
      --map-db-out /data/section01/track/map.msg

Outputs: `track/frame_trajectory.txt` (TUM), `track/keyframe_trajectory.txt`, `track/map.msg`.

NOTE (verify on first run): binary path inside the image and flag names can drift
between releases — run the container with `--help` first and correct this runbook:
    docker run --rm stella_vslam-socket /stella_vslam/build/run_video_slam --help
```

- [ ] **Step 6: Commit**

```powershell
git add pipeline/track3dgs/track.py pipeline/tests/test_track.py docs/runbooks/stella-vslam.md
git commit -m "feat: track stage - VSLAM ingest, scaling, per-view COLMAP model, QC plot"
```

---

### Task 7: Stage 4 — `cells`

**Files:**
- Create: `pipeline/track3dgs/cells.py`, `pipeline/tests/test_cells.py`

**Interfaces:**
- Consumes: `poses.json`, `track/colmap/` (via `load_images_txt`), `views_meta`; `write_colmap_model` (re-used for subsets — reconstruct `T_wc` per view from stored quat/trans by inverting `world_to_camera`).
- Produces: CLI `python -m track3dgs.cells --project DIR [--cell 50] [--pad 10] [--tile 10]`. Function `run_cells(project_dir, cell=50.0, pad=10.0, tile=10.0)`.
  - `plan_cells(total_s, cell, pad) -> list[dict]`: `{"id": n, "s_core": [n*cell, min((n+1)*cell, total_s)], "s_full": [max(0, n*cell - pad), min(total_s, (n+1)*cell + pad)]}` — last cell absorbs a remainder shorter than `cell/2` into the previous cell.
  - Output: `cells/cells.json` = `{"cell_length","pad","tile_length","total_s","cells":[…]}` where each cell also gets `"frames": [names of rig frames with s in s_full]`; per cell `cells/cell_{id:03d}/colmap/` model containing only that cell's view images (all yaws of its frames) + full `cameras.txt` + points3D filtered to the cell's XYZ bounding box expanded by 20 m.
  - `invert_wc(q, t) -> T_wc` helper (inverse of `trajectory.world_to_camera`) exported for reuse in tests.

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_cells.py`:
```python
import numpy as np
from track3dgs.cells import plan_cells, invert_wc, run_cells
from track3dgs.io_utils import Project, ensure_dir, read_json, write_json
from track3dgs.colmap_export import write_colmap_model, load_images_txt
from track3dgs.trajectory import world_to_camera


def test_plan_cells_padding_and_remainder():
    cells = plan_cells(112.0, cell=50.0, pad=10.0)
    assert len(cells) == 2                     # 12 m remainder < 25 -> absorbed
    assert cells[0]["s_core"] == [0.0, 50.0]
    assert cells[0]["s_full"] == [0.0, 60.0]
    assert cells[1]["s_core"] == [50.0, 112.0]
    assert cells[1]["s_full"] == [40.0, 112.0]


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
    for i in range(8):                          # rig frames every 10 m: s=0..70
        T = np.eye(4); T[:3, 3] = [10.0 * i, 0, 0]
        name = f"frame_{i:06d}.jpg"
        frames.append({"name": name, "t": float(i), "s": 10.0 * i,
                       "T_wc": [float(v) for v in T.reshape(-1)]})
        images.append({"name": f"frame_{i:06d}_y+090.jpg", "T_wc": T})
    write_json(p.poses_json, {"scale": 1.0, "frames": frames})
    pts = np.array([[5.0, 0, 2.0], [65.0, 0, 2.0]])
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_cells.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `cells.py`**

`pipeline/track3dgs/cells.py`:
```python
"""Stage 4: arc-length cell planning + per-cell COLMAP subsets (global frame)."""
import argparse

import numpy as np

from .colmap_export import load_images_txt, write_colmap_model
from .io_utils import Project, read_json, write_json
from .trajectory import quat_to_R


def plan_cells(total_s, cell=50.0, pad=10.0):
    n = max(1, int(total_s // cell))
    if total_s - n * cell >= cell / 2:
        n += 1
    cells = []
    for i in range(n):
        s0, s1 = i * cell, min((i + 1) * cell, total_s)
        if i == n - 1:
            s1 = total_s
        cells.append({"id": i, "s_core": [s0, s1],
                      "s_full": [max(0.0, s0 - pad), min(total_s, s1 + pad)]})
    return cells


def invert_wc(q, t):
    qw, qx, qy, qz = q
    R_cw = quat_to_R(qx, qy, qz, qw)
    T = np.eye(4)
    T[:3, :3] = R_cw.T
    T[:3, 3] = -R_cw.T @ np.asarray(t, dtype=float)
    return T


def _load_points3d(path):
    pts = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split()
            pts.append([float(f[1]), float(f[2]), float(f[3])])
    return np.array(pts) if pts else None


def run_cells(project_dir, cell=50.0, pad=10.0, tile=10.0):
    p = Project(project_dir)
    poses = read_json(p.poses_json)
    frames = poses["frames"]
    total_s = frames[-1]["s"]
    cells = plan_cells(total_s, cell, pad)

    meta = read_json(p.views_meta)
    all_imgs = load_images_txt(p.colmap_dir / "images.txt")
    points = _load_points3d(p.colmap_dir / "points3D.txt")

    for c in cells:
        lo, hi = c["s_full"]
        cframes = [f for f in frames if lo <= f["s"] <= hi]
        c["frames"] = [f["name"] for f in cframes]
        stems = {f["name"].rsplit(".", 1)[0] for f in cframes}
        cimgs = [{"name": im["name"], "T_wc": invert_wc(im["q"], im["t"])}
                 for im in all_imgs
                 if im["name"].rsplit("_y", 1)[0] in stems]
        cpts = None
        if points is not None and cframes:
            centers = np.array([np.array(f["T_wc"]).reshape(4, 4)[:3, 3]
                                for f in cframes])
            bb_lo, bb_hi = centers.min(0) - 20.0, centers.max(0) + 20.0
            inside = np.all((points >= bb_lo) & (points <= bb_hi), axis=1)
            cpts = points[inside] if inside.any() else None
        write_colmap_model(p.cells_dir / f"cell_{c['id']:03d}" / "colmap",
                           meta, cimgs, cpts)

    write_json(p.cells_json, {"cell_length": cell, "pad": pad,
                              "tile_length": tile, "total_s": total_s,
                              "cells": cells})
    print(f"{len(cells)} cells over {total_s:.1f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--cell", type=float, default=50.0)
    ap.add_argument("--pad", type=float, default=10.0)
    ap.add_argument("--tile", type=float, default=10.0)
    a = ap.parse_args()
    run_cells(a.project, a.cell, a.pad, a.tile)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_cells.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add pipeline/track3dgs/cells.py pipeline/tests/test_cells.py
git commit -m "feat: cells stage - arc-length cell planning and per-cell COLMAP subsets"
```

---

### Task 8: Stage 5 — `train` wrapper (+ Nerfstudio environment)

**Files:**
- Create: `pipeline/track3dgs/train.py`, `pipeline/tests/test_train.py`, `docs/runbooks/nerfstudio-setup.md`

**Interfaces:**
- Consumes: `cells/cells.json`, per-cell colmap dirs, `views/` + `views_masks/`.
- Produces: CLI `python -m track3dgs.train --project DIR --cell N [--iters 30000] [--cap 2500000] [--dry-run]`. Functions:
  - `build_train_cmd(project: Project, cell_id, iters, cap) -> list[str]` — the exact `ns-train` argv (below), unit-testable without GPU.
  - `build_export_cmd(config_yml, out_dir) -> list[str]` — `["ns-export", "gaussian-splat", "--load-config", str(config_yml), "--output-dir", str(out_dir)]`.
  - `check_alignment(exported_ply, cell_colmap_dir) -> dict` — loads exported PLY centroid + bbox vs `points3D.txt` bbox; returns `{"offset_m": float, "ok": bool}` with `ok = offset_m < 2.0`. Uses `plyfile`.
  - `run_train(project_dir, cell_id, iters, cap, dry_run)` — runs the two commands via `subprocess.run(check=True)`, moves/renames exported PLY to `train/export/cell_{id:03d}.ply`, runs `check_alignment`, fails loudly if not ok.
- The canonical argv (pose normalization OFF is the load-bearing part):
```
ns-train splatfacto
  --data <cells/cell_XXX>
  --output-dir <train>  --experiment-name cell_XXX  --timestamp run
  --max-num-iterations <iters>
  --viewer.quit-on-train-completion True
  --pipeline.model.strategy mcmc
  --pipeline.model.max-gs-num <cap>
  colmap
  --colmap-path colmap
  --images-path <abs path to views/>
  --masks-path <abs path to views_masks/>
  --center-method none  --orientation-method none  --auto-scale-poses False
  --load-3D-points True
```

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_train.py`:
```python
import numpy as np
from plyfile import PlyData, PlyElement
from track3dgs.io_utils import Project, ensure_dir
from track3dgs.train import build_train_cmd, build_export_cmd, check_alignment


def test_build_train_cmd_disables_normalization(tmp_path):
    p = Project(tmp_path / "sec")
    cmd = build_train_cmd(p, 0, iters=30000, cap=2_500_000)
    s = " ".join(cmd)
    assert "--center-method none" in s
    assert "--orientation-method none" in s
    assert "--auto-scale-poses False" in s
    assert "--pipeline.model.max-gs-num 2500000" in s
    assert str(p.cells_dir / "cell_000") in s


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
    good = tmp_path / "good.ply"; _write_ply(good, np.array([[14.0, 0, 0], [16.0, 0, 0]]))
    bad = tmp_path / "bad.ply"; _write_ply(bad, np.array([[500.0, 0, 0]]))
    assert check_alignment(good, colmap)["ok"]
    assert not check_alignment(bad, colmap)["ok"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_train.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `train.py`**

`pipeline/track3dgs/train.py`:
```python
"""Stage 5: per-cell Splatfacto training in the fixed global frame."""
import argparse
import shutil
import subprocess

import numpy as np
from plyfile import PlyData

from .io_utils import Project, ensure_dir


def build_train_cmd(project, cell_id, iters, cap):
    cell_dir = project.cells_dir / f"cell_{cell_id:03d}"
    return ["ns-train", "splatfacto",
            "--data", str(cell_dir),
            "--output-dir", str(project.train_dir),
            "--experiment-name", f"cell_{cell_id:03d}",
            "--timestamp", "run",
            "--max-num-iterations", str(iters),
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.model.strategy", "mcmc",
            "--pipeline.model.max-gs-num", str(cap),
            "colmap",
            "--colmap-path", "colmap",
            "--images-path", str(project.views_dir.resolve()),
            "--masks-path", str(project.views_masks_dir.resolve()),
            "--center-method", "none",
            "--orientation-method", "none",
            "--auto-scale-poses", "False",
            "--load-3D-points", "True"]


def build_export_cmd(config_yml, out_dir):
    return ["ns-export", "gaussian-splat",
            "--load-config", str(config_yml),
            "--output-dir", str(out_dir)]


def check_alignment(exported_ply, cell_colmap_dir, max_offset=2.0):
    v = PlyData.read(str(exported_ply))["vertex"]
    centroid = np.array([v["x"].mean(), v["y"].mean(), v["z"].mean()])
    pts = []
    for line in (cell_colmap_dir / "points3D.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split()
            pts.append([float(f[1]), float(f[2]), float(f[3])])
    ref = np.array(pts).mean(axis=0)
    offset = float(np.linalg.norm(centroid - ref))
    return {"offset_m": offset, "ok": offset < max_offset}


def run_train(project_dir, cell_id, iters=30000, cap=2_500_000, dry_run=False):
    p = Project(project_dir)
    cmd = build_train_cmd(p, cell_id, iters, cap)
    cfg = p.train_dir / f"cell_{cell_id:03d}" / "splatfacto" / "run" / "config.yml"
    exp = build_export_cmd(cfg, p.export_dir / f"cell_{cell_id:03d}")
    if dry_run:
        print(" ".join(cmd)); print(" ".join(exp)); return
    subprocess.run(cmd, check=True)
    subprocess.run(exp, check=True)
    src = p.export_dir / f"cell_{cell_id:03d}" / "splat.ply"
    dst = p.export_dir / f"cell_{cell_id:03d}.ply"
    ensure_dir(p.export_dir)
    shutil.move(str(src), str(dst))
    res = check_alignment(dst, p.cells_dir / f"cell_{cell_id:03d}" / "colmap")
    print(f"alignment offset {res['offset_m']:.2f} m -> "
          f"{'OK' if res['ok'] else 'FAILED (normalization leaked!)'}")
    if not res["ok"]:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--cap", type=int, default=2_500_000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run_train(a.project, a.cell, a.iters, a.cap, a.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_train.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write Nerfstudio setup runbook + install + flag verification**

`docs/runbooks/nerfstudio-setup.md`:
```markdown
# Nerfstudio + gsplat on RTX 5060 Ti (Blackwell sm_120)

    conda activate track3dgs
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install nerfstudio
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    # expect: True  NVIDIA GeForce RTX 5060 Ti

gsplat compiles CUDA kernels on first use (needs MSVC Build Tools + CUDA 12.8 toolkit
on PATH). Smoke test:
    python -c "import gsplat; print(gsplat.__version__)"

If the JIT build fails on Windows, fall back to WSL2 Ubuntu 22.04 with the same
commands (CUDA WSL driver required) — the pipeline is path-portable via /mnt/c.

FLAG VERIFICATION (do once, correct train.py if names differ in installed version):
    ns-train splatfacto --help | findstr strategy      # expect --pipeline.model.strategy {default,mcmc}
    ns-train splatfacto --help | findstr max-gs-num    # expect --pipeline.model.max-gs-num INT
    ns-train splatfacto colmap --help | findstr center-method
    ns-export gaussian-splat --help
Record installed versions here: nerfstudio==____  gsplat==____  torch==____
Also verify: nerfstudio mask convention is "white=keep / zero=ignore"
(docs.nerf.studio "Using custom data", masks section). If inverted, flip in views.py.
```

Run the installs and the four verification commands; fix `build_train_cmd` flag names if the installed version differs, and re-run `pytest tests/test_train.py`.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/track3dgs/train.py pipeline/tests/test_train.py docs/runbooks/nerfstudio-setup.md
git commit -m "feat: train stage - splatfacto wrapper with normalization disabled + alignment check"
```

---

### Task 9: Stage 6 — `slice`

**Files:**
- Create: `pipeline/track3dgs/slice.py`, `pipeline/tests/test_slice.py`

**Interfaces:**
- Consumes: `train/export/cell_XXX.ply`, `poses.json` (trajectory), `cells/cells.json`.
- Produces: CLI `python -m track3dgs.slice --project DIR --cell N [--max-dist 60] [--min-opacity 0.005]`. Functions:
  - `gaussian_s_values(xyz (N,3), traj_pts (M,3), traj_s (M,)) -> (s (N,), dist (N,))` — nearest dense-trajectory point per Gaussian via `scipy.spatial.cKDTree`; returns its arc-length and distance.
  - `assign_tiles(s, core_lo, core_hi, tile_len) -> np.ndarray (N,) int` — tile index `int(s // tile_len)`; Gaussians with `s < core_lo` or `s >= core_hi` get `-1` (pad zone — owned by the neighbouring cell).
  - `prune_mask(dist, opacity_raw, max_dist, min_opacity) -> bool mask` — keep if `dist <= max_dist` and `sigmoid(opacity_raw) >= min_opacity`.
  - `run_slice(project_dir, cell_id, max_dist, min_opacity)` — reads cell PLY, writes `tiles/tile_{idx:04d}.ply` per non-empty tile (all vertex properties preserved via numpy structured-array slicing), prints kept/pruned/pad counts.

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_slice.py`:
```python
import numpy as np
from plyfile import PlyData, PlyElement
from track3dgs.slice import gaussian_s_values, assign_tiles, prune_mask, run_slice
from track3dgs.io_utils import Project, ensure_dir, write_json


def test_gaussian_s_values_straight_line():
    traj = np.stack([np.arange(0, 101, 1.0), np.zeros(101), np.zeros(101)], axis=1)
    s = np.arange(0, 101, 1.0)
    xyz = np.array([[10.2, 0, 3.0], [55.0, 0, -7.0]])
    gs, gd = gaussian_s_values(xyz, traj, s)
    assert abs(gs[0] - 10.0) < 0.61 and abs(gd[0] - 3.0) < 0.1
    assert abs(gs[1] - 55.0) < 0.51 and abs(gd[1] - 7.0) < 0.1


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
    frames = [{"name": f"f{i}.jpg", "t": float(i), "s": float(i * 10),
               "T_wc": [float(v) for v in (np.eye(4) + 0).reshape(-1)]}
              for i in range(6)]
    for i, f in enumerate(frames):             # straight line along +x
        T = np.eye(4); T[:3, 3] = [i * 10.0, 0, 0]
        f["T_wc"] = [float(v) for v in T.reshape(-1)]
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
    assert "f_dc_0" in t1.data.dtype.names       # all props preserved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_slice.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `slice.py`**

`pipeline/track3dgs/slice.py`:
```python
"""Stage 6: slice a trained cell PLY into non-overlapping 10 m tiles + prune."""
import argparse

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from .io_utils import Project, ensure_dir, read_json


def gaussian_s_values(xyz, traj_pts, traj_s):
    tree = cKDTree(traj_pts)
    dist, idx = tree.query(xyz)
    return traj_s[idx], dist


def assign_tiles(s, core_lo, core_hi, tile_len):
    t = (s // tile_len).astype(int)
    t[(s < core_lo) | (s >= core_hi)] = -1
    return t


def prune_mask(dist, opacity_raw, max_dist, min_opacity):
    op = 1.0 / (1.0 + np.exp(-opacity_raw))
    return (dist <= max_dist) & (op >= min_opacity)


def run_slice(project_dir, cell_id, max_dist=60.0, min_opacity=0.005):
    from .trajectory import resample_polyline
    p = Project(project_dir)
    cj = read_json(p.cells_json)
    cell = next(c for c in cj["cells"] if c["id"] == cell_id)
    tile_len = cj["tile_length"]

    frames = read_json(p.poses_json)["frames"]
    positions = np.array([np.array(f["T_wc"]).reshape(4, 4)[:3, 3] for f in frames])
    traj_pts, traj_s = resample_polyline(positions, step=0.5)

    ply = PlyData.read(str(p.export_dir / f"cell_{cell_id:03d}.ply"))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(float)
    s, dist = gaussian_s_values(xyz, traj_pts, traj_s)
    keep = prune_mask(dist, v["opacity"].astype(float), max_dist, min_opacity)
    tiles = assign_tiles(s, cell["s_core"][0], cell["s_core"][1], tile_len)

    ensure_dir(p.tiles_dir)
    n_pad = int((tiles == -1).sum())
    for tid in sorted(set(tiles[keep & (tiles >= 0)])):
        sel = keep & (tiles == tid)
        el = PlyElement.describe(v[sel], "vertex")
        PlyData([el]).write(str(p.tiles_dir / f"tile_{tid:04d}.ply"))
        print(f"tile_{tid:04d}: {int(sel.sum())} splats")
    print(f"pruned {int((~keep).sum())} / pad-dropped {n_pad} of {len(v)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--max-dist", type=float, default=60.0)
    ap.add_argument("--min-opacity", type=float, default=0.005)
    a = ap.parse_args()
    run_slice(a.project, a.cell, a.max_dist, a.min_opacity)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_slice.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add pipeline/track3dgs/slice.py pipeline/tests/test_slice.py
git commit -m "feat: slice stage - arc-length tiling with pad ownership and pruning"
```

---

### Task 10: Stage 7 — `pack` (+ merged section PLY)

**Files:**
- Create: `pipeline/track3dgs/pack.py`, `pipeline/tests/test_pack.py`

**Interfaces:**
- Consumes: `tiles/tile_*.ply`, `cells/cells.json`, `poses.json`.
- Produces: CLI `python -m track3dgs.pack --project DIR [--merge]`. Functions:
  - `tile_record(ply_path, tile_len) -> dict` — `{"id": int, "file": name, "s_start": id*tile_len, "s_end": (id+1)*tile_len, "num_splats": int, "bounds": {"min":[x,y,z],"max":[x,y,z]}}`.
  - `run_pack(project_dir, merge=False)` — writes `tiles/manifest.json`: `{"version": 1, "tile_length", "total_s", "trajectory": [{"s", "pos":[x,y,z]} per rig frame], "tiles": [tile_record…]}`. With `merge=True` also concatenates all tile vertex arrays into `demo/section_merged.ply`.
- The manifest is the exact contract Unity's TileManager will consume in Phase 2.

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_pack.py`:
```python
import numpy as np
from plyfile import PlyData, PlyElement
from track3dgs.io_utils import Project, ensure_dir, write_json, read_json
from track3dgs.pack import tile_record, run_pack


def _ply(path, xs):
    v = np.zeros(len(xs), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    v["x"] = xs
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))


def _make_project(tmp_path):
    p = Project(tmp_path / "sec")
    ensure_dir(p.tiles_dir)
    _ply(p.tiles_dir / "tile_0000.ply", [1.0, 2.0])
    _ply(p.tiles_dir / "tile_0001.ply", [11.0])
    write_json(p.cells_json, {"cell_length": 50.0, "pad": 10.0,
                              "tile_length": 10.0, "total_s": 20.0, "cells": []})
    T = np.eye(4)
    write_json(p.poses_json, {"scale": 1.0, "frames": [
        {"name": "f.jpg", "t": 0.0, "s": 0.0,
         "T_wc": [float(v) for v in T.reshape(-1)]}]})
    return p


def test_tile_record(tmp_path):
    p = _make_project(tmp_path)
    r = tile_record(p.tiles_dir / "tile_0001.ply", 10.0)
    assert r["id"] == 1 and r["s_start"] == 10.0 and r["num_splats"] == 1
    assert r["bounds"]["min"][0] == 11.0


def test_run_pack_manifest_and_merge(tmp_path):
    p = _make_project(tmp_path)
    run_pack(p.root, merge=True)
    m = read_json(p.manifest_json)
    assert m["version"] == 1 and len(m["tiles"]) == 2
    assert m["tiles"][0]["num_splats"] == 2
    merged = PlyData.read(str(p.demo_dir / "section_merged.ply"))["vertex"]
    assert len(merged) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_pack.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `pack.py`**

`pipeline/track3dgs/pack.py`:
```python
"""Stage 7: tile manifest for Unity + optional merged section PLY for the demo."""
import argparse
import re

import numpy as np
from plyfile import PlyData, PlyElement

from .io_utils import Project, ensure_dir, read_json, write_json


def tile_record(ply_path, tile_len):
    tid = int(re.search(r"tile_(\d+)", ply_path.name).group(1))
    v = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    return {"id": tid, "file": ply_path.name,
            "s_start": tid * tile_len, "s_end": (tid + 1) * tile_len,
            "num_splats": int(len(v)),
            "bounds": {"min": [float(x) for x in xyz.min(0)],
                       "max": [float(x) for x in xyz.max(0)]}}


def run_pack(project_dir, merge=False):
    p = Project(project_dir)
    cj = read_json(p.cells_json)
    tile_len = cj["tile_length"]
    tile_files = sorted(p.tiles_dir.glob("tile_*.ply"))
    tiles = [tile_record(f, tile_len) for f in tile_files]

    frames = read_json(p.poses_json)["frames"]
    traj = [{"s": f["s"],
             "pos": [float(x) for x in np.array(f["T_wc"]).reshape(4, 4)[:3, 3]]}
            for f in frames]

    write_json(p.manifest_json, {"version": 1, "tile_length": tile_len,
                                 "total_s": cj["total_s"], "trajectory": traj,
                                 "tiles": tiles})
    print(f"manifest: {len(tiles)} tiles, "
          f"{sum(t['num_splats'] for t in tiles):,} splats total")

    if merge:
        arrays = [PlyData.read(str(f))["vertex"].data for f in tile_files]
        merged = np.concatenate(arrays)
        ensure_dir(p.demo_dir)
        PlyData([PlyElement.describe(merged, "vertex")]).write(
            str(p.demo_dir / "section_merged.ply"))
        print(f"merged: {len(merged):,} splats -> demo/section_merged.ply")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    run_pack(a.project, a.merge)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_pack.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add pipeline/track3dgs/pack.py pipeline/tests/test_pack.py
git commit -m "feat: pack stage - Unity tile manifest and merged demo PLY"
```

---

### Task 11: Demo camera path generator

**Files:**
- Create: `pipeline/track3dgs/demo_camera_path.py`, `pipeline/tests/test_demo_camera_path.py`

**Interfaces:**
- Consumes: `poses.json`.
- Produces: CLI `python -m track3dgs.demo_camera_path --project DIR [--fps 30] [--speed 4.0] [--fov 75]`. Function `build_camera_path(frames, fps, speed_mps, fov) -> dict` in Nerfstudio camera-path JSON shape:
  `{"camera_type": "perspective", "render_height": 1080, "render_width": 1920, "fps": fps, "seconds": total_s/speed, "camera_path": [{"camera_to_world": [16 floats row-major], "fov": fov, "aspect": 1.7777} …]}` — resamples rig poses at constant `speed_mps` along arc-length (position lerp + nearest-pose rotation), smooths positions with a 5-sample moving average. Writes `demo/camera_path.json`.
- NOTE: Nerfstudio's camera-path JSON uses **OpenGL-style camera axes in world space** as produced by its own viewer exports; on first real render, if the video looks backwards/upside-down, apply the axis flip `camera_to_world[:3,1:3] *= -1` (OpenCV→OpenGL) in `build_camera_path` — a `--flip-axes` flag (default True) is included for this.

- [ ] **Step 1: Write failing tests**

`pipeline/tests/test_demo_camera_path.py`:
```python
import numpy as np
from track3dgs.demo_camera_path import build_camera_path


def _frames_line(n, step):
    out = []
    for i in range(n):
        T = np.eye(4); T[:3, 3] = [i * step, 0, 0]
        out.append({"name": f"f{i}.jpg", "t": float(i), "s": i * step,
                    "T_wc": [float(v) for v in T.reshape(-1)]})
    return out


def test_build_camera_path_length_and_shape():
    frames = _frames_line(11, 2.0)             # 20 m
    cp = build_camera_path(frames, fps=10, speed_mps=4.0, fov=75.0,
                           flip_axes=False)
    assert cp["camera_type"] == "perspective"
    assert abs(cp["seconds"] - 5.0) < 1e-6      # 20 m / 4 mps
    assert len(cp["camera_path"]) == 50         # seconds * fps
    first = np.array(cp["camera_path"][0]["camera_to_world"]).reshape(4, 4)
    assert abs(first[0, 3]) < 1.0               # starts near s=0


def test_flip_axes_changes_rotation():
    frames = _frames_line(11, 2.0)
    a = build_camera_path(frames, 10, 4.0, 75.0, flip_axes=False)
    b = build_camera_path(frames, 10, 4.0, 75.0, flip_axes=True)
    Ra = np.array(a["camera_path"][0]["camera_to_world"]).reshape(4, 4)
    Rb = np.array(b["camera_path"][0]["camera_to_world"]).reshape(4, 4)
    assert np.allclose(Ra[:3, 1], -Rb[:3, 1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd pipeline; python -m pytest tests/test_demo_camera_path.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `demo_camera_path.py`**

`pipeline/track3dgs/demo_camera_path.py`:
```python
"""Investor demo: constant-speed flythrough camera path (nerfstudio JSON)."""
import argparse

import numpy as np

from .io_utils import Project, ensure_dir, read_json, write_json


def _smooth(positions, k=5):
    if len(positions) < k:
        return positions
    kernel = np.ones(k) / k
    sm = positions.copy()
    for i in range(3):
        sm[:, i] = np.convolve(np.pad(positions[:, i], k // 2, mode="edge"),
                               kernel, mode="valid")[:len(positions)]
    return sm


def build_camera_path(frames, fps, speed_mps, fov, flip_axes=True):
    s = np.array([f["s"] for f in frames])
    Ts = [np.array(f["T_wc"]).reshape(4, 4) for f in frames]
    positions = _smooth(np.array([T[:3, 3] for T in Ts]))
    total = float(s[-1])
    seconds = total / speed_mps
    n = int(round(seconds * fps))

    path = []
    for i in range(n):
        sq = total * i / max(1, n - 1)
        j = int(np.searchsorted(s, sq).clip(1, len(s) - 1))
        w = (sq - s[j - 1]) / max(1e-9, s[j] - s[j - 1])
        pos = positions[j - 1] * (1 - w) + positions[j] * w
        T = Ts[j - 1] if w < 0.5 else Ts[j]      # nearest rotation
        C = T.copy(); C[:3, 3] = pos
        if flip_axes:                            # OpenCV -> OpenGL camera axes
            C[:3, 1:3] *= -1
        path.append({"camera_to_world": [float(v) for v in C.reshape(-1)],
                     "fov": fov, "aspect": 1.7777})
    return {"camera_type": "perspective", "render_height": 1080,
            "render_width": 1920, "fps": fps, "seconds": seconds,
            "camera_path": path}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--fov", type=float, default=75.0)
    ap.add_argument("--no-flip-axes", action="store_true")
    a = ap.parse_args()
    p = Project(a.project)
    frames = read_json(p.poses_json)["frames"]
    cp = build_camera_path(frames, a.fps, a.speed, a.fov,
                           flip_axes=not a.no_flip_axes)
    ensure_dir(p.demo_dir)
    write_json(p.demo_dir / "camera_path.json", cp)
    print(f"{len(cp['camera_path'])} keyframes, {cp['seconds']:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline; python -m pytest tests/test_demo_camera_path.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full unit suite**

Run: `cd pipeline; python -m pytest -v`
Expected: all tests from Tasks 1–11 pass.

- [ ] **Step 6: Commit**

```powershell
git add pipeline/track3dgs/demo_camera_path.py pipeline/tests/test_demo_camera_path.py
git commit -m "feat: demo camera path generator (nerfstudio camera-path JSON)"
```

---

### Task 12: End-to-end runbook on the real test section

This task is executed by a human+agent pair at the console (GPU + visual QC steps). Every command is exact; QC gates are listed after each stage. Record all findings in `docs/runbooks/section01-log.md` as you go.

**Files:**
- Create: `docs/runbooks/section01-log.md` (running log, created during execution)

- [ ] **Step 1: Choose the section and run `extract`**

Scrub the source video, pick a 30–40 m stretch with steady motion, good light, no other vehicles. Note start/end seconds. Place source video at `data\raw\track.mp4`. Then:

```powershell
conda activate track3dgs
cd pipeline
python -m track3dgs.extract --video ..\data\raw\track.mp4 --out ..\data\section01 --start <S> --end <E>
```

QC gate (spec §8 stage-1 checks): open 5 spread-out files from `data\section01\frames\` — confirm sharp foliage, no gross stitching seams at the side yaws, steady exposure. Log frame count.

- [ ] **Step 2: Paint the vehicle mask**

Open `data\section01\mask_reference.jpg` in any editor (Paint/GIMP), paint the vehicle body, mounts, and any fixed rigging **black** on a white canvas of the same size; save as `data\section01\mask_equirect.png` (grayscale PNG). QC: overlay-check by eye that no vehicle pixel is white.

- [ ] **Step 3: Run `views`**

```powershell
python -m track3dgs.views --project ..\data\section01
```

QC gate (spec §8 stage-2 checks): inspect `views\` — forest centred in ±90° crops, straight trunks (no equirect bend), consecutive same-yaw crops overlap visibly (>60%). Inspect one mask crop per yaw.

- [ ] **Step 4: Run stella_vslam (Docker) then `track`**

Follow `docs/runbooks/stella-vslam.md` (build image once, verify binary/flags with `--help`, then run on `proxy_vslam.mp4`). Then, with measured or estimated vehicle speed:

```powershell
python -m track3dgs.track --project ..\data\section01 --speed-kmh <V>
```

QC gate (spec §8 stage-3 checks): open `track\qc_trajectory.png` — smooth curve, no jumps/teleports; printed `length` ≈ the chosen 30–40 m; if `frames dropped` warning exceeds 10%, re-run VSLAM with `max_num_keypoints: 3000`. Log scale factor.

- [ ] **Step 5: Run `cells`**

```powershell
python -m track3dgs.cells --project ..\data\section01
```

QC gate: `cells\cells.json` shows 1 cell covering the whole section (30–40 m < 50 m); its `frames` list ≈ all frames.

- [ ] **Step 6: Train the cell**

```powershell
python -m track3dgs.train --project ..\data\section01 --cell 0 --dry-run   # inspect argv first
python -m track3dgs.train --project ..\data\section01 --cell 0
```

Expect 30 k iterations (~20–60 min on the 5060 Ti). The built-in alignment check must print `OK` — if it prints `FAILED (normalization leaked!)`, the dataparser flags did not take; re-check `nerfstudio-setup.md` flag verification before proceeding.
QC gate (spec §8 stage-5 checks): during training open the viewer link, orbit near the trajectory — tree detail present, no giant floaters; note final splat count from the training log.

- [ ] **Step 7: Slice, pack, merge**

```powershell
python -m track3dgs.slice --project ..\data\section01 --cell 0
python -m track3dgs.pack --project ..\data\section01 --merge
```

QC gate (spec §8 stage-6 checks): printed per-tile splat counts roughly balanced (within 3×); total after pruning noted in the log (target ≤ ~400 K × tile count); pruned fraction expected 30–50 %.

- [ ] **Step 8: Tile-boundary inspection**

Open `data\section01\demo\section_merged.ply` in SuperSplat (https://superspl.at/editor — drag & drop). Fly along the track at trunk height across every 10 m boundary: no gaps, no doubled trunks, ground continuous (spec §8: boundary strip inspection). Then hide/show individual `tiles\tile_*.ply` (load them as separate splats in SuperSplat) to confirm each tile stands alone with a clean hard edge at its boundary.

- [ ] **Step 9: Commit the log**

```powershell
git add docs/runbooks/section01-log.md
git commit -m "docs: section01 end-to-end run log"
```

---

### Task 13: Investor demo deliverables

- [ ] **Step 1: Render the flythrough video**

```powershell
python -m track3dgs.demo_camera_path --project ..\data\section01 --speed 4.0
ns-render camera-path --load-config ..\data\section01\train\cell_000\splatfacto\run\config.yml --camera-path-filename ..\data\section01\demo\camera_path.json --output-path ..\data\section01\demo\flythrough.mp4
```

If the first rendered frames look backwards or upside-down, re-run `demo_camera_path` with `--no-flip-axes` (axis-convention check, see Task 11 note). Expected: 1080p mp4, camera gliding along the track at constant 4 m/s.

- [ ] **Step 2: Polish pass**

Re-render variants and pick the best for the investor audience: `--speed 2.5` (slower = more cinematic), `--fov 65`. Trim/encode the final cut:

```powershell
ffmpeg -i ..\data\section01\demo\flythrough.mp4 -c:v libx264 -crf 18 -pix_fmt yuv420p ..\data\section01\demo\flythrough_final.mp4
```

- [ ] **Step 3: Interactive viewer package**

For the live demo: open `section_merged.ply` in SuperSplat, verify smooth navigation, and save the SuperSplat scene settings. Fallback if offline is required: `ns-viewer --load-config ...\config.yml` on the demo laptop.

- [ ] **Step 4: Acceptance checklist (spec §8)**

- [ ] Flythrough video: no floaters crossing the camera path, foliage stable, no visible tile seams.
- [ ] Interactive viewer: 30+ fps navigation near the track on the demo machine.
- [ ] Held-out sanity: pick 3 rig frames spread along the section — for each, render the trained model from that frame's +90° view pose (`ns-render` single frame or viewer screenshot) and place it side-by-side with the real crop `views/frame_XXXXXX_y+090.jpg` in the log.
- [ ] Log final numbers in `section01-log.md`: total splats, per-tile counts, training time, VRAM peak.

- [ ] **Step 5: Commit**

```powershell
git add docs/runbooks/section01-log.md
git commit -m "docs: investor demo acceptance results"
```

---

## Self-Review Notes

- **Spec coverage:** stages 1–7 (spec §4) → Tasks 2,3,6,7,8,9,10; global-frame discipline (spec §2) → Task 8 flags + `check_alignment`; scale-without-GPS (spec §5) → Task 6 `compute_scale_from_speed/known`; vehicle mask (spec §7) → Tasks 2,3,12; investor demo (spec §3 Phase 1) → Tasks 11,13; validation items (spec §8) → QC gates in Tasks 12–13. Unity runtime, Quest budget measurement, and ODGS are Phase 2/0/3 — intentionally out of scope here.
- **Known external-tool uncertainty (by design, with verification steps):** stella_vslam CLI flags (Task 6 runbook `--help` check), nerfstudio flag names + mask convention (Task 8 Step 5 verification), nerfstudio camera-path axis convention (Task 11 `--flip-axes` + Task 13 Step 1 check). These are verify-then-correct steps, not placeholders — the expected values are written down and the correction procedure is defined.
- **Type consistency check:** `view_name` (Task 3) used by Tasks 6; `world_to_camera`/`load_images_txt` round-trip via `invert_wc` (Task 7); `poses.json` schema identical in Tasks 6,9,10,11; `cells.json` schema identical in Tasks 7,9,10.
