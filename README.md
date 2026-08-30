# Track3DGS

**360° vehicle-mounted track video → chunked 3D Gaussian Splatting → streamed in Unity on Quest 3.**

A 12 km forest track is recorded once with a roof-mounted 360° camera (8K equirect, 30 fps).
This repo reconstructs it as 3DGS models divided into **10 m runtime tiles**, which a Unity
app on Meta Quest 3 streams while a simulated vehicle drives the track — only ~3 tiles
resident at a time. Visual quality only; no collision, viewer stays on the vehicle.

- **Design spec:** [`docs/superpowers/specs/2026-08-06-track3dgs-pipeline-design.md`](docs/superpowers/specs/2026-08-06-track3dgs-pipeline-design.md)
- **Implementation plan (historic reference):** [`docs/superpowers/plans/2026-08-06-phase1-offline-pipeline.md`](docs/superpowers/plans/2026-08-06-phase1-offline-pipeline.md)

## Core design decisions

1. **Train big, slice small.** Training happens per ~50 m *cell*; trained Gaussians are
   bucketed by trajectory arc-length into non-overlapping 10 m *tiles* for streaming.
2. **One fixed global metric frame, everywhere.** Trajectory, sparse cloud, trained
   cells, and tiles share one coordinate system. Training runs with pose normalization
   disabled; tiles butt together seamlessly — no runtime blending, only show/hide.
3. **Every Gaussian belongs to exactly one tile.** Overlapping-chunk blending was
   rejected: independently trained overlapping sets double-render and shimmer.
4. **Equirect → pinhole crops.** 8 yaw directions (±45/±90/±135 + 0/180 bridges,
   100° FOV) give perfect pinhole images with known intrinsics. The front/rear bridge
   views are load-bearing: without them SfM splits into disconnected left/right models.
5. **Mask, don't reconstruct, what isn't scene.** The vehicle hull (hand-painted once
   per rig) and the sky (Mask2Former per frame + brightness backstop) are excluded from
   feature extraction and training loss. Sky comes from a Unity skybox later.
6. **Metric scale from average vehicle speed** (no GPS). Single global factor,
   correctable post-hoc.
7. **Orientation via one-time mount calibration, not estimation.** Auto up-estimators
   (camera axis, ground RANSAC) proved unreliable in corridor forest scenes. Instead the
   camera-mount-to-gravity relationship was measured once from a manual SuperSplat
   leveling ([`pipeline/mount_calibration.json`](pipeline/mount_calibration.json)) and
   every section is leveled automatically against it (`level` stage).
8. **Validate with physical constraints.** All 8 views of a frame share one optical
   centre → mis-registered views are provably detectable and auto-dropped (`track`).
   Splats must have COLMAP-point support and mask-consistent projections to survive
   (`slice`, `skyprune`).

## The 7 steps per section

1. **Cut & extract** — cut the section in LosslessCut → `data\raw\sectionNN.mp4`, then
   `extract` pulls sharp frames. *Smoke test:* browse `frames\`.
2. **Vehicle mask** — copy `mask_equirect.png` from a previous section (rig constant)
   or hand-paint once. *Smoke test:* `qc_vehicle_mask.jpg` (red = excluded).
3. **Sky mask** — `skymask` (Mask2Former + brightness backstop), per frame.
   *Smoke test:* `qc_sky\*.jpg` overlays.
4. **Views + COLMAP** — `views` (8 pinhole yaws, combined masks) then `track`
   (masked features, sequential matching, mapper) with automatic rig-consistency
   validation (flier views dropped) and point quality/radial filters.
   *Smoke test:* registration %, `qc_trajectory.png`, COLMAP GUI on `track\colmap`
   (the cleaned model — set Render options → min track length 0).
5. **Level (fix orientation)** — `level` rotates the whole frame to the calibrated
   orientation from `pipeline/mount_calibration.json`; origin at trajectory start.
   *Smoke test:* `qc_leveled.png` — flat camera line, upright corridor cross-section.
6. **Train 3DGS** — `cells` then `train` (Splatfacto, pose normalization off, masked
   loss); export auto-undoes nerfstudio's frame rotation; the NN-distance alignment
   guard fails loudly on any frame leak (expect ~0.06 m when healthy).
   *Smoke test:* the guard number, then the exported PLY in SuperSplat.
7. **Prune & polish** — `skyprune` applies three automatic tiers: mask-projection
   sky dome, colour-assisted canopy glitter, and needle-spike removal (one dominant
   scale axis >0.5 m at >=8x anisotropy, or >5 m outright; flat road/wall splats
   survive by construction). Then optional manual cleanup in SuperSplat (start/end
   junk); finally `slice` + `pack` produce the 10 m tiles + Unity manifest.
   *Smoke test:* `cell_XXX_skyremoved.ply` (audit what was deleted), tile toggling.

## Pipeline stages (command reference, run per section from `pipeline/`)

```powershell
$py  = "..\.venv\Scripts\python.exe"          # utility venv (py 3.14, no CUDA)
$pyt = "..\.venv-train\Scripts\python.exe"    # training venv (py 3.11 + CUDA)
# cuda_env.bat wraps a command with MSVC v143 + CUDA 12.8 + UTF-8 (needed for train/skymask)

& $py  -m track3dgs.extract  --video ..\data\raw\sectionNN.mp4 --out ..\data\sectionNN
#   -> sharp frames (best-of-3), proxy video, mask template. QC: frames\, sharpness stats
Copy-Item ..\data\section01\mask_equirect.png ..\data\sectionNN\   # rig mask is reusable
#   QC: overlay (red = masked) -> qc_vehicle_mask.jpg
.\cuda_env.bat $pyt -m track3dgs.skymask  --project ..\data\sectionNN --model mask2former
#   -> per-frame sky masks. QC: qc_sky\*.jpg overlays (red = sky)
& $py  -m track3dgs.views    --project ..\data\sectionNN --yaws "-135,-90,-45,0,45,90,135,180"
#   -> 8 pinhole crops per frame + combined vehicle+sky masks per view
& $py  -m track3dgs.track    --project ..\data\sectionNN --speed-kmh <V> --overlap 48
#   -> COLMAP (masked features) + rig-consistency validation + 60 m radial point filter
#   QC: registration %, track\qc_trajectory.png, COLMAP GUI (set Render options min track length 0)
& $py  -m track3dgs.level    --project ..\data\sectionNN
#   -> whole frame rotated to the calibrated orientation, origin at trajectory start
#   QC: track\qc_leveled.png (flat camera line), residual printout
& $py  -m track3dgs.cells    --project ..\data\sectionNN
.\cuda_env.bat $pyt -m track3dgs.train --project ..\data\sectionNN --cell 0
#   -> Splatfacto 30k iters on masked crops; export auto-undoes nerfstudio's
#      COLMAP->OpenGL rotation; alignment guard fails loudly if any frame leak
& $py  -m track3dgs.skyprune --project ..\data\sectionNN --cell 0
#   -> two-tier sky cleanup: mask-projection (dome) + color-assist (canopy glitter)
#   QC: load cell_XXX_skyremoved.ply in SuperSplat to audit removals
& $py  -m track3dgs.slice    --project ..\data\sectionNN --cell 0
#   -> 10 m tiles; support-distance, road-protection, end-cut pruning
& $py  -m track3dgs.pack     --project ..\data\sectionNN --merge
#   -> pre-leveled passthrough, SH strip (Quest), manifest.json + merged demo PLY
# Manual polish: edit in SuperSplat, export; positions can be fingerprint-matched back
```

Tests: `python -m pytest` from `pipeline/` (50+, no GPU needed).

## Calibration files (precious)

- `pipeline/mount_calibration.json` — camera-axes directions in a level world; measured
  once from a manual SuperSplat leveling, valid for the whole track (rig constant).
  Re-derive only if the camera is remounted.
- `data/section01/demo/leveling_calibration.json` — the raw measured rotation from the
  user's leveled+edited export (fingerprint Kabsch, 0.0 mm residual); source of the above.
- `data/sectionNN/mask_equirect.png` — hand-painted vehicle mask (white=keep). Reusable
  across sections; back it up.

## Known open issues

- **SuperSplat export-vs-import transform**: SuperSplat bakes a coordinate change on
  export that it does not apply on import, so calibrations measured from *exports* are
  flipped (~180°) relative to what *loads* level. Fix pending: one no-edit round-trip
  export to measure the bake exactly, then fold into the mount key.
- **Residual canopy glitter**: `skyprune` removes the sky dome and much treetop glitter;
  remnants persist (conservative thresholds). A StableGS-class trainer was evaluated
  (`gstrain.py`: gsplat MCMC cap + opacity reg + sparse-depth supervision) but lost the
  visual A/B against Splatfacto + skyprune (2026-08-29); it remains available as an
  experimental backend and as the only route to a training-time splat cap.
- ns-export uses nerfstudio's internal frame; `train.py` undoes it (`NS_EXPORT_FIX`,
  verified against `dataparser_transforms.json`).
- Windows: plyfile mmap blocks in-place rewrite (handled); nerfstudio needs UTF-8 env
  (handled in `cuda_env.bat`); background installers may silently die on missed UAC.
- **Patched file in .venv-train** (reapply after any nerfstudio reinstall):
  `nerfstudio/data/datamanagers/full_images_datamanager.py` — pin_memory falls back to
  unpinned RAM when the page-locked limit is exceeded (sections >~1000 views OOM'd).

## Environments & tools

| What | Where | Notes |
|---|---|---|
| Utility venv | `.venv` (py 3.14) | all pipeline stages except train/skymask |
| Training venv | `.venv-train` (py 3.11) | torch 2.7.1+cu128, nerfstudio 1.1.5, gsplat 1.4.0 (JIT-compiled sm_120), transformers |
| Build env | `pipeline/cuda_env.bat` | MSVC v143 (14.44) + CUDA 12.8 + UTF-8 + venv PATH |
| COLMAP 4.1.1 | `C:\Work\tools\colmap` | GUI via `COLMAP.bat`, not bin\colmap.exe |
| LosslessCut | `C:\Work\tools\LosslessCut` | keyframe-exact section cutting |
| SuperSplat | https://superspl.at/editor | viewing, leveling, manual polish |

## Section status

| Section | Extract | Masks | COLMAP | Level | Train | Notes |
|---|---|---|---|---|---|---|
| section01 (34.2 m) | ✅ 124 frames | ✅ | ✅ 992/992, 16 views auto-dropped | ✅ | ✅ 922 K splats, sky-pruned to 909 K | orientation fix pending SuperSplat bake measurement |
| section02 (~50 m) | ✅ 179 frames | ✅ | queued | — | — | needs `--speed-kmh` |

## Setup on a new machine

See the environments table; in short: python 3.11+ and 3.14 venvs per
`pipeline/requirements.txt` + the training stack (torch cu128 → nerfstudio → pinned
`fpsample==0.3.3` wheel → gsplat via JIT with MSVC v143 + CUDA 12.8), FFmpeg on PATH,
COLMAP binaries. `data/` is **not in git** — copy the raw videos, masks, and calibration
files manually. Full war stories in the git log.
