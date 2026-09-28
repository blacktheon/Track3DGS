# 03 — Pipeline Steps Reference

Every stage of the reconstruction pipeline: command, inputs, outputs, formats, and
worked examples. All commands run **from the `pipeline/` directory**. The pipeline is
the Python package `track3dgs`; each stage is a module you invoke with `-m`.

## Two Python interpreters (see doc 04)

- `..\.venv\Scripts\python.exe` — utility env (no GPU). CPU stages.
- `..\.venv-train\Scripts\python.exe` — GPU env (torch+CUDA+nerfstudio+gsplat).
  **Must be launched through `cuda_env.bat`** which sets MSVC + CUDA 12.8 + UTF-8 on
  PATH. Used by `skymask` (Mask2Former), `train`, `qc`/`evalpsnr` (they render).

Shorthand used below: `$py` = utility python, `$pyt` = `cuda_env.bat <train python>`.

## The "project" concept

Every section is a **project directory** under `data/` (e.g. `data/t01-1_s02`). All
stages read and write through the `Project` class (`track3dgs/io_utils.py`), which
defines the canonical file layout. You pass `--project <dir>`; the stage knows where
everything goes. The full layout of a finished project:

```
data/t01-1_s02/
  section.mp4              trimmed section video (t=0 at section start)
  proxy_vslam.mp4          1920x960 proxy (legacy, for stella_vslam; unused by COLMAP)
  mask_equirect.png        the vehicle mask for this section (copied in at step 2)
  mask_reference.jpg       first frame, for hand-painting a mask
  frames/                  selected sharp 8K equirect frames: frame_000000.jpg ...
  frames_meta.jsonl        one JSON object per kept frame (see §1)
  sky_masks/               per-frame sky masks: frame_000000.png ... (white=keep,black=sky)
  views/                   perspective crops: frame_000000_y+090.jpg ... + views_meta.json
  views_masks/             per-view combined vehicle+sky masks: frame_000000_y+090.png ...
  track/
    poses.json             per-frame rig pose + arc-length + scale (see §4)
    colmap/                COLMAP text model: cameras.txt, images.txt, points3D.txt
    colmap_work/           COLMAP database + raw binary model (deletable after QC)
    qc_trajectory.png      top-down + height-profile plot
    qc_leveled.png         post-level elevation/cross-section plot
    poses_oldframe.json    (only if a re-ingest happened) backup of pre-level poses
  cells/
    cells.json             cell plan (see §6)
    cell_000/colmap/       per-cell COLMAP subset (the training input)
  train/
    cell_000/              nerfstudio run output (config.yml, checkpoints)
    export/
      cell_000.ply         raw trained model (full SH, ~240 MB)
      cell_000_skypruned.ply   final model after 3-tier pruning (the deliverable)
      cell_000_skyremoved.ply  audit file: ONLY the splats that were removed
  tiles/                   (only if slice/pack was run — NOT run for Track01)
    tile_0000.ply ...      10 m tiles
    packed/                gravity-baked, SH-stripped tiles for Unity
    manifest.json          Unity streaming manifest (see §10)
  timing.csv               per-step wall-clock (written by run_section.ps1)
```

---

## Step 1 — `extract`: video → sharp frames

**Command**
```powershell
& $py -m track3dgs.extract --video ..\data\raw\track01-1\section01.mp4 --out ..\data\t01-1_s01
```
Args: `--start`/`--end` (seconds, default whole file), `--group N` (keep best of every
N frames, default 3), `--proxy-width` (default 1920).

**Input**: an mp4 section (already cut, see §orchestration). 8K equirect, 30 fps.

**What it does**: re-encodes/copies the section so t=0 is the start; extracts all
frames; scores each by **variance-of-Laplacian** sharpness; keeps the sharpest of every
group of 3 (motion-blur rejection); builds the 1920×960 proxy; writes a mask template.

**Outputs**:
- `frames/frame_NNNNNN.jpg` — the kept frames, numbered by *source* frame index.
- `frames_meta.jsonl` — one object per line:
  ```json
  {"name": "frame_000005.jpg", "src_index": 5, "t": 0.1667, "sharpness": 1234.5}
  ```
  `t` = seconds from section start = `src_index / fps`.
- `mask_reference.jpg`, `mask_equirect.png` (all-white template; real mask copied in at
  step 2).

**Smoke test**: browse `frames/` — sharp foliage, no gross motion blur.

---

## Step 2 — vehicle mask (a copy, not a computation)

**Command** (this is what `run_section.ps1` does):
```powershell
Copy-Item ..\data\raw\track01_vehicle_mask.png (Join-Path $Project "mask_equirect.png") -Force
```

**Input/Output**: the canonical hand-painted mask
`data/raw/track01_vehicle_mask.png` (7680×3840, **white=keep world, black=vehicle**) is
copied to `<project>/mask_equirect.png`. One mask serves all Track01 sections because
the footage is body-locked (doc 02 §F1).

**QC overlay** (optional): the mask tinted red over the reference frame →
`qc_vehicle_mask.jpg`. Verify red covers the hull and nothing else.

*(There is also `automask.py`, a temporal-variance auto-mask generator, used once to
bootstrap; the hand-painted mask supersedes it. `automask` refuses to overwrite an
existing mask without `--force`.)*

---

## Step 3 — `skymask`: per-frame sky masks

**Command** (GPU env — Mask2Former runs on GPU):
```powershell
$pyt -m track3dgs.skymask --project ..\data\t01-1_s01 --model union --prior ..\data\raw\track01_sky_prior.png
```
`--model` ∈ `union` (default, production), `mask2former`, `prior`, `color`, `segformer`.

**Input**: `frames/` + the sky prior `data/raw/track01_sky_prior.png` (hand-drawn:
black = "sky is possible here"). The union model needs the prior; pure `mask2former`
does not.

**What it does** (union): `Mask2Former(ADE20K sky) OR (prior AND blue-to-white colour
test AND enclosed-hole fill)`. See doc 02 §F2 for the full rationale.

**Output**: `sky_masks/frame_NNNNNN.png` — grayscale, **white=keep, black=sky**
(same convention as the vehicle mask). Prints sky-fraction stats (mean ~35%).

**Smoke test**: overlays in `qc_sky/*.jpg` (red = will be excluded). All sky/cloud
tinted, no real foliage/road tinted.

---

## Step 4a — `views`: equirect frames → 8 pinhole crops

**Command**
```powershell
& $py -m track3dgs.views --project ..\data\t01-1_s01 --yaws "-135,-90,-45,0,45,90,135,180"
```
Args: `--fov` (default 100), `--size` (default 1600).

**Input**: `frames/`, `mask_equirect.png`, `sky_masks/`.

**What it does**: for each frame and each yaw, renders a 1600² pinhole crop
(`py360convert`) and the matching **combined mask** (vehicle AND sky, cropped the same
way). The 8 yaws include the load-bearing 0°/180° bridge views (doc 02 §B2).

**Outputs**:
- `views/frame_NNNNNN_y{+iii}.jpg` — e.g. `frame_000000_y+090.jpg`, `..._y-045.jpg`.
  8 per frame.
- `views/views_meta.json` — the shared pinhole intrinsics + yaw list:
  ```json
  {"width":1600,"height":1600,"fov_deg":100.0,"fx":671.3,"fy":671.3,
   "cx":800.0,"cy":800.0,"yaws":[-135,-90,-45,0,45,90,135,180]}
  ```
- `views_masks/frame_NNNNNN_y{+iii}.png` — the combined mask per view.

**Smoke test**: side views show sharp forest, straight trunks; the hull is masked in
lower views.

---

## Step 4b — `track`: COLMAP SfM → global poses + sparse cloud

**Command**
```powershell
& $py -m track3dgs.track --project ..\data\t01-1_s01 --speed-kmh 10 --overlap 48 --mapper glomap
```
`--mapper` ∈ `glomap` (default, the built-in `global_mapper`) or `colmap` (incremental
fallback). `--ingest-only` re-runs only the pose/scale ingest on an existing model.
`--colmap` points at the exe (default `C:\Work\tools\colmap\bin\colmap.exe`).

**Input**: `views/`, `views_masks/`, `views_meta.json`.

**What it does**: COLMAP feature extraction (masked — vehicle+sky features ignored) →
sequential matching (48-neighbour window) → global mapper → then our *ingest*:
- **rig-consistency validation** drops flying-camera views (doc 02 §C3);
- recovers the per-frame vehicle pose from its registered views;
- computes **metric scale** from `--speed-kmh` × duration ÷ arc-length;
- **radial-filters** points >60 m from the trajectory.

**Outputs**:
- `track/poses.json`:
  ```json
  {"scale": 2.0404,
   "frames": [
     {"name":"frame_000000.jpg","t":0.0,"s":0.0,
      "T_wc":[16 floats, row-major 4x4 camera-to-world]},
     ...]}
  ```
  `s` = arc-length in metres; `T_wc` = rig pose. **Convention: OpenCV/COLMAP camera
  (x-right, y-down, z-forward); `T_wc` is camera-to-world.**
- `track/colmap/{cameras,images,points3D}.txt` — the scaled COLMAP model.
- `track/qc_trajectory.png` — top-down path + height profile.

**Smoke test**: registration ratio (want ~100%), `qc_trajectory.png` smooth, and the
COLMAP GUI for a fly-around:
```powershell
C:\Work\tools\colmap\COLMAP.bat gui --database_path <proj>\track\colmap_work\database.db `
  --image_path <proj>\views --import_path <proj>\track\colmap
```
*(In the GUI set Render options → min track length 0, or the simplified points are
hidden and you see only cameras — a known confusion, not a bug.)*

---

## Step 5 — `level`: rotate the frame upright (mount calibration)

**Command**
```powershell
& $py -m track3dgs.level --project ..\data\t01-1_s01
```
`--mount` overrides the key file (default `pipeline/mount_calibration.json`).

**Input**: `track/poses.json`, `track/colmap/`, and the mount key.

**What it does**: reads `mount_calibration.json` (the camera up-axis + travel direction
in a level world), computes the rotation that maps *this* section's camera axes onto the
key, and rewrites poses + COLMAP points already levelled, origin at the trajectory start.
Hands-free (doc 02 §D2).

**Output**: `track/poses.json` and `track/colmap/` rewritten in place (adds
`"leveled": true`); `track/qc_leveled.png`. Prints `cam-up residual 0.00 deg`.

**Smoke test**: `qc_leveled.png` — flat camera line, upright corridor cross-section.

---

## Step 6a — `cells`: plan cells + per-cell COLMAP subset

**Command**
```powershell
& $py -m track3dgs.cells --project ..\data\t01-1_s01
```
Args: `--cell` (core length, default 50), `--pad` (default 10), `--tile` (default 10).

**What it does**: partitions the trajectory by arc-length into ~50 m cells (+10 m pad
each side). For each cell, writes the COLMAP subset (only that cell's views + nearby
points) that training consumes. For a short section this is **one cell = the whole
section**.

**Output**: `cells/cells.json`:
```json
{"cell_length":50.0,"pad":10.0,"tile_length":10.0,"total_s":24.72,
 "cells":[{"id":0,"s_core":[0.0,24.72],"s_full":[0.0,24.72],
           "frames":["frame_000000.jpg", ...]}]}
```
and `cells/cell_000/colmap/{cameras,images,points3D}.txt`.

---

## Step 6b — `train`: Splatfacto training → PLY

**Command** (GPU env)
```powershell
$pyt -m track3dgs.train --project ..\data\t01-1_s01 --cell 0
```
Args: `--iters` (default 30000), `--dry-run` (print the ns-train command only),
`--export-only` (skip training, just re-export+guard an existing run).

**What it does**: runs `ns-train splatfacto` on the cell with **pose normalization
disabled** and the combined masks applied to the loss; then `ns-export gaussian-splat`;
then applies `NS_EXPORT_FIX` (undo nerfstudio's hidden rotation, doc 02 §D4); then the
**nearest-neighbour alignment guard** (doc 02 §E1) — fails loudly on any frame leak.

**Output**: `train/export/cell_000.ply` (raw, full SH). Prints `alignment offset
0.06 m -> OK`.

**Smoke test**: load `cell_000.ply` in SuperSplat — upright, sharp foliage, no hull.

---

## Step 7 — `skyprune`: 3-tier artifact removal

**Command**
```powershell
& $py -m track3dgs.skyprune --project ..\data\t01-1_s01 --cell 0
```
Args: `--threshold` (sky-projection fraction, default 0.6), `--max-range` (default 50).

**What it does**: sky-dome (mask-projection) + canopy-glitter (colour) + needle-spike
(anisotropy) removal — doc 02 §G.

**Outputs**:
- `train/export/cell_000_skypruned.ply` — **the deliverable model.**
- `train/export/cell_000_skyremoved.ply` — audit file: only the removed splats. Load it
  alone in SuperSplat to see exactly what was deleted.
Prints e.g. `pruned 17,714 of 932,960 (1.9%): 11,904 sky + 3,145 glitter + 2,665
needle`.

---

## Steps 8 & 9 — export marker + QC (orchestration only)

`run_section.ps1` finishes with:
- **Step 8 export**: copy `cell_000_skypruned.ply` → `data/Export/<ExportName>.ply`.
  The presence of this file is the batch's "done" marker (doc 02 §H1).
- **Step 9 QC** (GPU env): `qc.py` prints the report and appends a row to
  `data/Export/qc_log.csv`:
  ```
  section,views_registered,rig_frames,length_m,guard_m,guard_ok,
  splats_trained,splats_final,pruned_pct,psnr_db
  t01-1_s02,720/720,90,24.7,0.056,True,965769,950322,1.6,20.49
  ```
  `psnr_db` = masked PSNR of the final model re-rendered at real view poses.

---

## NOT YET RUN ON TRACK01: `slice` + `pack` (tiling for Unity)

These exist and are unit-tested but were **not** part of the Track01 batch (Track01
deliverables are whole-section models). They are the bridge to Unity streaming.

**`slice`** (`$py -m track3dgs.slice --project <proj> --cell 0`) — dices the cell PLY
into non-overlapping 10 m `tiles/tile_NNNN.ply` by arc-length, with support-distance +
road-protection + end-cut pruning (doc 02 §G).

**`pack`** (`$py -m track3dgs.pack --project <proj> --merge`) — gravity-bakes and
SH-strips the tiles into `tiles/packed/`, and writes `tiles/manifest.json`, the Unity
streaming contract:
```json
{"version":1, "tile_length":10.0, "total_s":24.7,
 "coordinate_convention":"...", "sh_bands_stripped":true,
 "unity_import_euler":[...],
 "alignment":{"rotation":[9 floats],"translation":[3 floats]},
 "trajectory":[{"s":0.0,"pos":[x,y,z]}, ...],
 "tiles":[{"id":0,"file":"tile_0000.ply","s_start":0.0,"s_end":10.0,
           "num_splats":176364,"bounds":{"min":[..],"max":[..]}}, ...]}
```
This manifest is what a future Unity tile-streaming manager would consume (doc 05).

---

## Orchestration

**One section, end-to-end, timed** — `pipeline/run_section.ps1` runs steps 1→9 with
per-step timing to `<project>/timing.csv`. Parameters: `-Video`, `-Project`,
`-SpeedKmh` (required); `-Mapper`, `-MaskFrom`, `-SkyModel`, `-SkyPrior`, `-ExportName`,
`-Overlap` (defaulted to Track01 production values).
```powershell
.\run_section.ps1 -Video ..\data\raw\track01-1\section03.mp4 -Project ..\data\t01-1_s03 `
  -SpeedKmh 10 -ExportName Track01-1-S3
```

**A whole track, unattended** — `pipeline/run_track01_batch.ps1` iterates all sections
of track01-1/-2/-3, skips any already in `data/Export/`, continues past failures, and
writes `data/batch_track01_status.csv`. This is the script that ran 22.5 h with 26/26
success. To resume after a stop: just run it again.

**Cutting sections from raw video** — `pipeline/cuts_track01.ps1` holds the
owner-curated section timetable (start/end seconds per section) and produces the
lossless keyframe cuts under `data/raw/track01-*/`. Cuts are pure stream-copy
(no re-encode); the ~2 s overlaps between sections are intentional (future stitching
anchors).
