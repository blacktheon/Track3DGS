# Track3DGS

**360° vehicle-mounted track video → chunked 3D Gaussian Splatting → streamed in Unity on Quest 3.**

A 12 km forest track is recorded once with a roof-mounted 360° camera (8K equirect, 30 fps).
This repo reconstructs it as 3DGS models divided into **10 m runtime tiles**, which a Unity
app on Meta Quest 3 streams while a simulated vehicle drives the track — only ~3 tiles
resident at a time. Visual quality only; no collision, viewer stays on the vehicle.

- **Design spec:** [`docs/superpowers/specs/2026-08-06-track3dgs-pipeline-design.md`](docs/superpowers/specs/2026-08-06-track3dgs-pipeline-design.md)
- **Implementation plan (reference):** [`docs/superpowers/plans/2026-08-06-phase1-offline-pipeline.md`](docs/superpowers/plans/2026-08-06-phase1-offline-pipeline.md)

## Core design decisions

1. **Train big, slice small.** Training happens per ~50 m *cell* (better context, 10×
   fewer jobs); the trained Gaussians are then bucketed by trajectory arc-length into
   non-overlapping 10 m *tiles* for runtime streaming.
2. **One fixed global metric frame, everywhere.** The camera trajectory, the sparse
   cloud, every trained cell, and every tile share one coordinate system. Training runs
   with pose normalization *disabled*, so exported PLYs need no re-alignment, and tiles
   butt together seamlessly — no runtime blending, no cross-fade, only show/hide.
3. **Every Gaussian belongs to exactly one tile.** Overlapping-chunk blending was
   rejected: independently trained overlapping Gaussian sets double-render and shimmer.
4. **Equirect → pinhole crops before anything else.** 360° "distortion" is a known map
   projection; resampling 8 yaw directions (±45/±90/±135 + 0/180 bridges, 100° FOV)
   yields perfect pinhole images with exactly known intrinsics. The front/rear bridge
   views are load-bearing: without them SfM splits into disconnected left/right models.
5. **The vehicle hull is masked, not reconstructed.** A hand-painted equirect mask
   (white=world, black=vehicle) is carried through to every crop and excluded from
   feature extraction and training loss.
6. **Metric scale from average vehicle speed** (no GPS needed). A single global scale
   factor; correctable post-hoc without retraining.

## Phases

| Phase | Content | Status |
|-------|---------|--------|
| **1** | Offline reconstruction pipeline, proven on a 30–40 m test section | **in progress** (sub-projects 1–3 of 5 done) |
| **0** | Quest 3 rendering feasibility spike (splat budget, renderer choice) | after Phase 1 |
| **2** | Unity runtime tile streaming (3-tile window, show/hide) | after Phase 0 |
| **3** | Scale-up to 12 km (≈240 cells, batch over RTX 5060 Ti + DGX Spark) + polish | last |

### Phase 1 sub-projects (each ends in a visual smoke test)

| # | Sub-project | Tooling | Status / result |
|---|---|---|---|
| 1 | Section Cut | LosslessCut (`C:\Work\tools\LosslessCut`) | ✅ 12.4 s / ~34 m section cut losslessly |
| 2 | Frames & Views | `extract`, `views`, `automask` stages | ✅ 124 sharp frames → 992 crops + masks |
| 3 | Trajectory | `track` stage (COLMAP backend) | ✅ **992/992 views registered**, 34.2 m, 368 K points |
| 4 | First Splat | `cells` + `train` (Nerfstudio/gsplat) | next |
| 5 | Tiler | `slice` + `pack` → SuperSplat inspection | pending |

## Repository layout

```
pipeline/               Python package: one CLI per pipeline stage
  track3dgs/
    io_utils.py         shared on-disk project layout (class Project)
    extract.py          stage 1: trim video, sharp-frame selection, proxy, mask template
    views.py            stage 2: equirect -> 8 pinhole crops + per-view masks + intrinsics
    automask.py         optional: temporal-variance vehicle mask (hand-painted mask wins)
    trajectory.py       pose math: quats, TUM, yaw view poses, arc-length (OpenCV/COLMAP convention)
    colmap_export.py    COLMAP text model reader/writer
    track.py            stage 3: COLMAP feature/match/map -> scaled global poses + QC plot
  tests/                pytest suite (pure-math + synthetic-data tests, no GPU needed)
docs/superpowers/       design spec + implementation plan
data/                   NOT in git: raw video + all derived per-section data
```

### Per-section data layout (`data/section01/`)

```
section.mp4            trimmed section (t=0 at section start)
proxy_vslam.mp4        1920x960 proxy (for future stella_vslam use)
frames/                selected sharp 8K equirect frames
frames_meta.jsonl      {name, src_index, t, sharpness}
mask_equirect.png      hand-painted vehicle mask (white=keep) - PRECIOUS, back it up
views/                 992 pinhole crops frame_XXXXXX_y{+yaw}.jpg + views_meta.json
views_masks/           per-view mask PNGs (same stem, .png)
track/poses.json       per-frame rig pose T_wc (4x4, row-major) + arc-length s + scale
track/colmap/          scaled COLMAP text model (all 992 views + sparse points)
track/qc_trajectory.png  top-down + height-profile plot
track/colmap_work/     COLMAP database + raw binary models (can be deleted after QC)
```

## Setup on a new machine (Windows)

1. **Python 3.11+** (3.14 tested), **FFmpeg** on PATH (`winget install Gyan.FFmpeg`).
2. ```powershell
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r pipeline\requirements.txt
   ```
3. **COLMAP 4.x** Windows binaries → `C:\Work\tools\colmap`
   (or pass `--colmap <path>` to the track stage). GUI needs `COLMAP.bat`, not `bin\colmap.exe`.
4. **LosslessCut** (section cutting UI) → `C:\Work\tools\LosslessCut` (optional).
5. Copy the `data/` folder (it is gitignored — transfer manually, especially
   `data/raw/*.mp4` and any hand-painted `mask_equirect.png`).
6. Sanity check: `cd pipeline; ..\.venv\Scripts\python.exe -m pytest` → all green, no GPU needed.

Sub-project 4 additionally needs the CUDA stack (PyTorch cu128 + Nerfstudio + gsplat) —
see the plan document, Task 8.

## Running the pipeline on a section

```powershell
cd pipeline
$py = "..\.venv\Scripts\python.exe"
& $py -m track3dgs.extract  --video ..\data\raw\section01.mp4 --out ..\data\section01
& $py -m track3dgs.automask --project ..\data\section01     # or hand-paint mask_equirect.png
& $py -m track3dgs.views    --project ..\data\section01 --yaws "-135,-90,-45,0,45,90,135,180"
& $py -m track3dgs.track    --project ..\data\section01 --speed-kmh 10
# -- sub-project 4+ (upcoming) --
# & $py -m track3dgs.cells  --project ..\data\section01
# & $py -m track3dgs.train  --project ..\data\section01 --cell 0
# & $py -m track3dgs.slice  --project ..\data\section01 --cell 0
# & $py -m track3dgs.pack   --project ..\data\section01 --merge
```

Visual QC after each stage: inspect `frames/`, `views/`, `mask_overlay.jpg`,
`track/qc_trajectory.png`, and open the sparse model in the COLMAP GUI:

```powershell
C:\Work\tools\colmap\COLMAP.bat gui `
  --database_path data\section01\track\colmap_work\database.db `
  --image_path data\section01\views `
  --import_path data\section01\track\colmap_work\sparse\0
```

## Section01 reference numbers (for regression comparison)

- Source: 7680×3840 HEVC 29.97 fps, 12.38 s (371 frames), vehicle ≈ 10 km/h
- `extract`: 124/371 frames kept (best-of-3 sharpness)
- `views`: 992 crops (8 yaws × 124), fx = 671.3 px, hand-painted mask keeps 62.3 %
- `track` (COLMAP, fixed PINHOLE intrinsics, sequential overlap 48):
  992/992 registered, single model, 368,551 points, scale 2.9283, length 34.2 m,
  ~75 min wall-clock on RTX 5060 Ti
- Known trade-off: COLMAP here is the no-Docker fallback; stella_vslam (equirect,
  video-rate) is the intended backend for the 12 km batch (needs WSL2/Docker or DGX Spark).

## Notes & gotchas

- **Camera convention** everywhere: OpenCV/COLMAP (x-right, y-down, z-forward);
  `poses.json` stores camera-to-world; COLMAP `images.txt` stores world-to-camera.
- The world is **not gravity-aligned** yet (COLMAP picks an arbitrary orientation);
  alignment to Unity's Y-up happens in the pack stage (Phase 1 SP5 / Phase 2).
- `automask` refuses to overwrite an existing `mask_equirect.png` without `--force`.
- Re-running `views` regenerates crops *and* per-view masks; re-run it after any mask change.
- 8K JPEG decode is the throughput bottleneck of `views` (~2 it/s) — fine at test scale.
