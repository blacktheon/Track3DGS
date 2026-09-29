# Track3DGS

> **Current redesign (2026-09-29):** Track3DGS owns continuous-route reconstruction and regional training/assembly. [VR3DGS](https://github.com/blacktheon/VR3DGS) owns general manual reduction and offline LOD generation; the consuming application owns runtime streaming and renderer selection. Read the [system design](docs/architecture/pipeline-integration.md), [portable package contract](docs/contracts/asset-package-v1.md), and [reuse/modification plan](docs/superpowers/plans/2026-09-29-route-reconstruction-and-regions.md). The new [Step 1 route workflow](docs/route-step1.md) implements bounded continuous-video ingestion, shared camera coordinates, training-region proposals and Unity marker export. Dense regional training, seam validation and package export remain planned work.
>
> The technical inventory below preserves the working section pipeline and its experimental history. Its global coordinates apply within each separately processed section; use the opt-in route workflow for a continuous capture. The older fixed 10 m tiles / nearest-three design is historical. Standalone Quest performance has not been established by the PC/Link tests.

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

## Techniques, models & algorithms (detailed inventory)

**Capture & preparation**
- **LosslessCut** — keyframe-snapped stream-copy section cutting (zero re-encode).
- **FFmpeg** — CFR section normalization, frame extraction, VSLAM proxy downscale.
- **Variance-of-Laplacian sharpness scoring** with best-of-N group selection —
  motion-blur rejection (keeps 1 of every 3 frames at 30 fps).

**Masking**
- Hand-painted equirect **vehicle mask** (rig constant, one per camera mount);
  an automatic **temporal-variance mask** (per-pixel std over frames + morphology)
  exists as a starting point (`automask`).
- **Mask2Former (Swin-Large, ADE20K)** semantic segmentation — per-frame sky masks
  (class 2); **SegFormer-b2** available as the lighter alternative.
- **HSV brightness backstop** — blown-out sun/cloud pixels above the horizon band
  forced to sky (segmentation misses saturated whites).
- **Area-gated dilation** (connected-components) — safety margin around the sky dome
  without eating foliage around small canopy gaps.

**Structure-from-Motion (step 4)**
- **Equirectangular→pinhole resampling** (`py360convert`) — 8-yaw virtual camera rig
  (±45/±90/±135 + 0/180 bridge views), exact analytic **PINHOLE intrinsics** (fixed,
  never refined).
- **COLMAP 4.1.1**: GPU **SIFT** with mask-excluded regions, **sequential matching**
  (48-neighbour window), and the **global SfM pipeline** (`global_mapper`, GLOMAP
  lineage): **rotation averaging → track establishment → global positioning →
  iterative bundle adjustment (Ceres) → retriangulation**. ~3x faster than the
  **incremental mapper** (kept as fallback) with 100% registration on all sections.
- **Rig-consistency validation** — all 8 views of a frame share one optical centre;
  views deviating >0.5 m from the frame median are provably mis-registered and
  dropped automatically.
- Point hygiene: **track-length ≥ 3**, **reprojection error ≤ 2 px**, **60 m radial
  filter** (cKDTree distance-to-trajectory).
- **Speed-based metric scaling** — average vehicle speed x duration / reconstructed
  arc-length (no GPS needed).

**Orientation (step 5)**
- **Mount calibration**: the camera-axes-to-gravity relation measured once from a
  manual SuperSplat leveling, recovered via **attribute-fingerprint matching +
  Kabsch/Procrustes rigid solve** (0.0 mm residual), then applied to every section by
  **two-vector triad alignment** (camera-up + travel direction). **RANSAC plane
  fitting** retained as a diagnostic (defeated by vegetation walls as an estimator).

**3DGS training (step 6)**
- **Nerfstudio Splatfacto** on the **gsplat CUDA rasterizer** — masked **L1 + SSIM**
  photometric loss, adaptive densification, **pose normalization disabled** so
  training stays in the leveled global metric frame.
- **Export frame fix** — inverse of nerfstudio's COLMAP→OpenGL `applied_transform`
  baked into every export (verified against `dataparser_transforms.json`).
- **Alignment guard** — median **nearest-neighbour distance** (cKDTree) from sampled
  splats to COLMAP points; ~0.07 m healthy, metres on any frame leak. Chosen after
  median-offset metrics proved density-biased.
- Evaluated and rejected by A/B: custom **gsplat MCMC trainer** with **opacity
  regularization** and **sparse-depth supervision** (StableGS/TIDI-GS-class
  mechanisms) — kept as experimental backend (`gstrain`), and the only route to a
  training-time splat cap.

**Artifact removal (step 7)**
- **2D→3D mask lifting** (FlashSplat-style, training-free): every Gaussian centre
  projected into all equirect sky masks via rig poses; ≥60% sky hits ⇒ sky dome.
- **Colour-assisted glitter pruning** — SH0 base colour rules (blue/blown-white) AND
  partial sky projection AND canopy height ⇒ treetop sparkle, road/verges immune.
- **Needle-spike pruning** — scale-anisotropy shape test (dominant axis >0.5 m with
  ≥8x mid-axis ratio, or >5 m outright); flat discs (roads/walls) survive by
  construction.
- **Support-distance pruning** — splats >2.5 m from any COLMAP point are unsupported
  (sky shells, underground mirror-fluff), with an 8 m trajectory-protection zone for
  the textureless road; **end-cut** drops content >10 m beyond the section ends.

**Packaging & runtime prep**
- **Arc-length tiling** — Gaussians bucketed into non-overlapping 10 m tiles by
  nearest-trajectory-point arc-length; pad-zone ownership rules; per-tile PLYs +
  **Unity manifest JSON** (bounds, counts, trajectory, transforms,
  `unity_import_euler`).
- **SH band stripping** (SH0) for rotation-exact transforms and Quest memory (~3x).

**Evaluation & QC**
- **Masked PSNR** — gsplat re-rendering at registered poses vs real crops.
- **Kabsch trajectory comparison** across pose engines / calibration chains.
- Per-stage visual QC artifacts (overlays, elevation projections, corridor
  cross-sections) + **COLMAP GUI** / **SuperSplat** inspection.

**Unity integration**
- **UnityGaussianSplatting** (aras-p) — requires **DirectX 12** in-editor (wave
  intrinsics for GPU sorting), **Vulkan** on the Android/Quest target, and the
  **GaussianSplatURPFeature** on every URP renderer asset. Import constant:
  rotation **(180, 0, 0)**.

## The 7 steps per section

1. **Cut & extract** — cut the section in LosslessCut → `data\raw\sectionNN.mp4`, then
   `extract` pulls sharp frames. *Smoke test:* browse `frames\`.
2. **Vehicle mask** — copy `mask_equirect.png` from a previous section (rig constant)
   or hand-paint once. *Smoke test:* `qc_vehicle_mask.jpg` (red = excluded).
3. **Sky mask** — `skymask` (Mask2Former + brightness backstop), per frame.
   *Smoke test:* `qc_sky\*.jpg` overlays.
4. **Views + COLMAP** — `views` (8 pinhole yaws, combined masks) then `track`
   (masked features, sequential matching, **global mapper** — default since the
   2026-08-31 A/B: ~3x faster than incremental, 100% registration, zero fliers,
   0.9 cm trajectory agreement; `--mapper colmap` remains the fallback) with
   automatic rig-consistency validation and point quality/radial filters.
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

## Unity editor preview (verified working)

Project needs: **DX12** editor graphics API (wave intrinsics for the splat sorter),
**Vulkan** for the Android/Quest target, `UnityGaussianSplatting` package, and the
**GaussianSplatURPFeature added to every URP renderer asset** in use (PC + Mobile).
Import full-SH exports (`cell_XXX_skypruned.ply`) via Tools → Gaussian Splats →
Create GaussianSplatAsset; set the GameObject rotation to **(180, 0, 0)** — the
verified Unity import constant, also recorded in each tile manifest.

## Track01 production setup (current)

- **Footage**: body-locked re-exports from `.insv` (FlowState/direction-lock OFF —
  stabilized exports rotate the hull in-frame on turns and break the static mask).
  Rear-facing original orientation kept: a 180-yaw "inverse" re-export was A/B'd and
  found geometrically identical (0.12 deg / 3.6 cm between independent runs) with a
  slight quality edge to the original (one fewer encode generation).
- **Section cutting**: user-curated timetable in `pipeline/cuts_track01.ps1` -
  27 sections across 3 videos, ~2 s designed overlaps, lossless keyframe cuts
  (integer-second keyframes, zero snap loss).
- **Sky masking**: `--model union` — Mask2Former OR (hand-drawn sky prior AND tuned
  blue-to-white colour test AND enclosed-cloud fill). Union beats either alone
  (34.0% coverage vs 33.7/32.6) at +20 ms/frame. Prior lives in
  `data/raw/track01_sky_prior.png`; for a 180-yaw export, masks transfer by a
  half-width `np.roll` (see `*_inverse.png` variants).
- **Batch**: `pipeline/run_track01_batch.ps1` runs all remaining sections
  unattended (~48 min per ~25 m section; mapper ~11 min at 720 views — short
  sections make global SfM cheap). Final models are copied to `data/Export/`
  as `TrackNN-V-SN.ply` after skyprune (step 8).

## Calibration files (precious)

- `pipeline/mount_calibration.json` — camera-axes directions in a level world;
  re-measured 2026-09-03 for the body-locked exports from the user's SuperSplat
  leveling of Track01-1-S1 (fingerprint Kabsch, 0.10 mm residual; the previous
  stabilized-era key is kept as `mount_calibration_stabilized_backup.json`).
  Re-derive only if the camera is remounted or the export style changes.
- `data/raw/track01_vehicle_mask.png` — hand-painted vehicle mask for the
  body-locked exports (99.5% static-hull coverage, verified across all 3 videos).
- `data/raw/track01_sky_prior.png` — hand-drawn sky-candidate zone for the union
  sky rule (black = sky possible).
- `data/Export/` — final per-section deliverables (leveled, pruned, optionally
  hand-polished in SuperSplat).

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
