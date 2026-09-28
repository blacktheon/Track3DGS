# 02 — Decisions & Lessons

This is the memory of the project. Each entry: **what was decided, why, and what was
tried and rejected.** Read it before changing anything — most of these were expensive
to learn. Loosely chronological, grouped by theme. Git commit hashes are cited so you
can see the actual change.

---

## A. Architecture (settled at design time)

**A1. "Train big, slice small" — cells then tiles, NOT per-chunk with blending.**
The original brainstorm considered training each 10 m chunk independently and
cross-fading neighbours at runtime. Rejected: independently-trained overlapping
Gaussian sets interleave in the depth sort, double up opacity, and never agree on
geometry — visible shimmer exactly where the viewer looks. Decision: train ~50 m cells
in one global frame, then slice into non-overlapping 10 m tiles. Every Gaussian belongs
to exactly one tile. (Spec §2.)

**A2. One fixed global metric frame; pose normalization OFF.** Nerfstudio/Splatfacto by
default re-centres and re-scales the scene. We disable that
(`--center-method none --orientation-method none --auto-scale-poses False`) so every
model stays in the same metric coordinates and tiles need no re-alignment.

**A3. Metric scale from vehicle speed, not GPS.** Monocular SfM has no absolute scale.
We recover one global scale factor from average vehicle speed × duration ÷
reconstructed arc-length. GPS was never required. *Caveat:* the driver held ~10 km/h
but real speed varied 5–15 km/h, so each section's absolute scale carries ±uncertainty.
This is reconciled later at the stitching stage (not yet built) via the 2 s overlaps.

**A4. Phase order changed to 1 → 0 → 2 → 3.** The spec originally put the Quest
feasibility spike first. The owner wanted a visible impressive result early (for
investors), so Phase 1 (reconstruction) went first. Consequence that paid off: the
demo is desktop SuperSplat fly-throughs at full quality, not gated on Quest work.

---

## B. Equirect → pinhole, and the SfM front end

**B1. Resample 8 perspective "views" per frame; do NOT feed equirect to COLMAP.**
COLMAP has no equirect camera model. We convert each 360° frame into 8 virtual pinhole
crops (`py360convert`, 100° FOV, 1600²) with exactly-known intrinsics. This is standard
and gives COLMAP clean, distortion-free, known-intrinsic images.

**B2. The front/rear "bridge" yaws are load-bearing.** First attempt used 6 yaws
(±45/±90/±135). Result: COLMAP split into 8 disconnected one-sided models and only
363/744 views registered. Adding **0° and 180°** (front and rear, looking down the
road corridor) bridged left and right — jumped to **992/992 registered, one connected
model**. The production yaw set is `-135,-90,-45,0,45,90,135,180`. Commit `e71bf81`.
*Do not drop the bridge views.*

**B3. Equirect "distortion" is not lens distortion.** It is a known analytic map
projection with uniform angular resolution in every direction. A tangential concern —
"is detail lost toward the back of the frame?" — was tested directly (doc: inverse
export A/B, §F3): front and back carry identical detail. The perspective resample
undoes the projection exactly; straight world lines come out straight.

---

## C. Camera trajectory (the `track` stage)

**C1. COLMAP on Windows, not stella_vslam/Docker.** stella_vslam (equirect visual SLAM,
video-rate) was the spec's intended engine for 12 km, but it is Linux/Docker and needs
WSL2 + admin. For the test sections COLMAP native Windows binaries gave the *most
accurate* trajectory with *zero install friction*, so COLMAP became the working engine.
stella_vslam remains the future option for the full 12 km (doc 05).

**C2. Global mapper (GLOMAP-style), not incremental — the default since A/B.** COLMAP's
default *incremental* mapper is O(n²)-ish and took ~2.5–3 h for a 1400-view section.
COLMAP 4.1.1 ships a built-in **`global_mapper`** (the GLOMAP pipeline: rotation
averaging → global positioning → bundle adjustment). Measured A/B on identical input:
**~3× faster mapper, 2.1× faster end-to-end, 100% registration, 0.9 cm trajectory
agreement** vs incremental, PSNR within noise. Adopted as default. Commit `5ee7e6b`.
Note: the *external* GLOMAP binary was tried first and failed on a database-schema
mismatch (`566c5a0`); the built-in `global_mapper` is what we use. `--mapper colmap`
keeps the incremental fallback.

**C3. Rig-consistency validation — automatic flying-camera removal.** All 8 views of a
frame are crops of one 360° photo, so they must share one optical centre. Any view
whose solved position deviates >0.5 m from its frame-siblings' median is provably
mis-registered and is dropped automatically. Catches ~10–16 views/section, almost
always the `y-135` rear-oblique (the hardest direction). Commit `0fa81be`.

**C4. Point hygiene filters.** Sparse points are kept only if seen by ≥3 images, have
reprojection error ≤2 px (`1cec2b6`), and lie within 60 m of the trajectory (`0fa81be`).
Removes sky-plug points and distant low-parallax noise before they seed training.

---

## D. Orientation — the hardest problem in the project

The camera's coordinate frame is arbitrary after SfM; the world comes out tilted. This
fought back repeatedly.

**D1. Auto up-estimators failed.** Tried: (a) average camera "up" axis — works only if
the mount is level and the stabilizer isn't lying; (b) RANSAC ground-plane normal —
**defeated by the vegetation walls**, which are large planar point sets parallel to the
road, so RANSAC locked onto a wall and produced an ~85° error (`f25d6c2` implemented it,
later abandoned as an estimator). SuperSplat convention quirks made hand-tuned Euler
angles a moving target too.

**D2. The solution: a one-time mount calibration.** Because the camera is rigidly
mounted, its axes have a *fixed* relationship to gravity — we just have to *measure* it
once. Procedure: run one section, the owner levels the result by eye in SuperSplat and
exports it, and we recover the exact rotation by **attribute-fingerprint matching +
Kabsch/Procrustes** (match splats between the owner's export and ours by their
colour/opacity signature — invariant to rotation — then solve the rigid rotation;
residual 0.1 mm). That rotation is distilled into `pipeline/mount_calibration.json`
(the camera up-axis and travel direction in a level world). Every subsequent section
auto-levels by aligning its own camera axes to that key (`level` stage, `0fa81be`,
`26e1404`). **This is why levelling is now hands-free.**

**D3. The mount key is per-mount / per-export-style.** It was re-measured once when the
footage was re-exported "body-locked" (§F1) because that changed the camera axes;
the old key is kept as `mount_calibration_stabilized_backup.json`. If the camera is
ever remounted or re-exported differently, re-measure (one SuperSplat levelling).

**D4. Nerfstudio bakes a hidden export rotation.** `ns-export gaussian-splat` applies a
COLMAP→OpenGL axis change (recorded in the run's `dataparser_transforms.json`) that it
does *not* undo. This silently broke orientation for ages. Fixed by applying its exact
inverse (`NS_EXPORT_FIX`) to every export (`26e1404`). The alignment guard (§E1) exists
partly to catch this class of frame leak.

---

## E. Training and its guards

**E1. Alignment guard via nearest-neighbour distance.** After training+export, we check
the model is still in the global frame by measuring the **median nearest-neighbour
distance from sampled splats to the COLMAP points** (~0.06 m when healthy, metres on any
frame leak). An earlier version compared cloud *centroids* but that was biased by
density differences and gave false failures; the NN-distance version is robust
(`1d1416c`). This guard is what makes unattended batches trustworthy — it fails loudly.

**E2. Splatfacto stays; the custom MCMC trainer lost its A/B.** A StableGS/TIDI-GS-class
custom trainer was built on gsplat directly — MCMC densification with a hard splat cap,
opacity regularization, sparse-depth supervision from the COLMAP points (`a0831d1`,
module `gstrain.py`). It trained fine but **lost the visual A/B against stock
Splatfacto + our pruning** (`b5e9f85`). Kept as an experimental backend and as the only
route to a *training-time* splat cap, but production is Splatfacto. Lesson: a
battle-tested trainer beat a hand-rolled one; don't rebuild the trainer to chase
quality.

**E3. Nerfstudio 1.1.5 has no MCMC/splat-cap flags.** The plan assumed they existed;
they don't in this version (`8650f0a`). The Quest splat budget is therefore enforced
*after* training, in the pruning/slicing stages, not during training.

---

## F. The two big data-quality discoveries

**F1. Stabilization broke the vehicle mask — re-export "body-locked".** The first
footage was exported with Insta360 **FlowState stabilization / direction-lock ON**. The
stabilizer rotates the equirect frame to hold the horizon, so on turns the *vehicle
hull moves within the frame* — and the vehicle mask (painted in fixed pixel space) no
longer covers it. The fix: re-export from the raw `.insv` files with **stabilization
and direction-lock OFF** ("body-locked"), so the hull sits in identical pixels for the
entire 12 km, turns included. Verified: hull pixel-stable across all three Track01
videos. This is why one hand-painted mask now serves the whole track.
*Anything the pipeline consumes must be a body-locked export.*

**F2. Sky masking evolved through four generations.** This took real iteration:
1. **SegFormer/Mask2Former (ADE20K "sky" class).** Good but region-level: it stops at
   the canopy silhouette and leaves blue gaps *inside* the tree crowns unmasked — those
   became "treetop glitter" splats. Also missed pale clouds/sun-glare (added an HSV
   **brightness backstop**, `9b822ce`) and its dilation ate leaf edges (fixed with
   **area-gated dilation** — only dilate the big sky dome, not small canopy gaps).
2. **Owner's insight — a colour rule.** Bright-day footage, nothing blue/white below
   the treeline, so: per-pixel blue-to-white test above a hand-drawn spatial prior.
   Pixel-precise at canopy gaps (masks the blue between leaves, keeps the leaves).
3. **Tuned** to spare grey rock and green-ish foliage (narrower blue hue band, require
   saturation, "pale" must be near-clipping *and* cold-toned B≥R), plus
   **enclosed-hole fill** (a non-sky island fully surrounded by sky inside the prior
   zone is a cloud → fill it; tree crowns touch the treeline so they survive).
4. **UNION (production default).** `Mask2Former OR (prior + colour + fill)`. Each covers
   the other's blind spots: the model gives the semantic treeline, the colour rule gives
   canopy-gap precision. Measured better coverage than either alone (34.0% vs
   33.7/32.6%) at +20 ms/frame (~0.4% of section time). Commit `3284c9f`. Selectable
   via `skymask --model {union|mask2former|prior|color|segformer}`.

*The colour/prior rule is weather-specific (bright day, blue sky). For overcast/other
conditions keep Mask2Former. That's why both live in the union and both remain
available.*

**F3. Inverse (180° yaw) export A/B — no benefit.** The owner wondered if the
rear-facing framing lost detail and re-exported rotated 180°. Tested end-to-end:
**geometrically identical** (0.12° / 3.6 cm between independent runs), masks transfer by
a half-width `np.roll`, quality within noise (the inverse was actually 0.35 dB *lower* —
one extra encode generation). Decision: **keep the original orientation**, drop the
inverse. Lesson banked: export direction is free to choose on workflow grounds.

---

## G. Artifact removal (the `skyprune` stage — 3 tiers)

Post-training cleanup, all automatic, all in `skyprune.py`:
1. **Sky-dome removal (mask projection / 2D→3D lifting, FlashSplat-style).** Project
   every splat centre into all the sky masks via the rig poses; if it lands on masked
   sky from ≥60% of nearby views, it's sky — delete it. Geometric, not colour-based.
   `ff64cf1`.
2. **Canopy-glitter removal (colour-assisted).** SH0 base colour is sky-blue/blown-white
   AND it has partial sky-projection evidence AND it sits above canopy height → treetop
   sparkle. Road and verges are structurally immune.
3. **Needle-spike removal (shape/anisotropy).** 3DGS overfits under-observed edges with
   extremely elongated Gaussians ("needles"). Prune where one scale axis > 0.5 m and
   ≥8× the second axis, or > 5 m outright. Flat discs (road, walls) have two large axes
   and survive by construction. `701415d`. (Research-backed — this is the standard
   remedy; see the session's web search on needle artifacts.)

There is also **support-distance pruning** in the `slice` stage (splats >2.5 m from any
COLMAP point are unsupported → sky shell / underground fluff; but an 8 m
trajectory-protection zone keeps the textureless road) and an **end-cut** (drop content
>10 m beyond the section ends). `b653267`, `f25d6c2`.

---

## H. Batch operations & Windows gotchas

**H1. Batch resume via Export markers.** The batch runner treats the presence of
`data/Export/<name>.ply` as a section's completion certificate and skips it on
relaunch; an interrupted section is cleanly rebuilt because each stage deletes its own
stale partial state first (COLMAP workspace, cells, train dirs). You can stop and
restart the batch at any time. `59dba7a`.

**H2. Per-section QC gate.** After every section, `qc.py` prints and appends to
`data/Export/qc_log.csv`: registration, alignment-guard distance + pass/fail, splat
counts, prune %, masked PSNR. A failed guard warns loudly and the batch continues, so
you get a list of anything to inspect rather than a stalled queue.

**H3. Windows-specific fixes that will bite you if reverted:**
- **UTF-8**: nerfstudio's rich console output crashes under cp1252; `cuda_env.bat` sets
  `PYTHONUTF8=1` (`3a4a124`).
- **plyfile mmap**: plyfile memory-maps on read, and Windows can't truncate a mapped
  file, so in-place PLY rewrites fail unless the mapping is released (`del ply; gc`)
  first (`5e1c1cb`).
- **torch 2.7 unpickling**: `ns-export` needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
  (`e687738`).
- **nerfstudio pin_memory OOM**: sections >~1000 views exhaust the page-locked pool and
  crash the CUDA context; a patch to `full_images_datamanager.py` in `.venv-train`
  skips pinning for large sets. **This patch lives in the venv, not git — reapply it
  after any nerfstudio reinstall.** (README "Known open issues".)

---

## I. Unity integration lessons

**I1. Three settings make splats render at all.** (a) Editor graphics API must be
**DirectX 12** (the splat sorter needs GPU wave intrinsics; DX11 fails). (b) The
Android/Quest target must be **Vulkan**. (c) The **`GaussianSplatURPFeature` must be
added to every URP renderer asset** in use — without it URP renders nothing and the
scene looks empty with no error. All three were needed before anything appeared.

**I2. Multi-camera conflict.** The tank sim has 8 cameras — 3 VR eye cameras (to screen)
and 5 crew-station periscope cameras (to 256×256 render textures). The splat feature ran
on all of them and threw a per-frame dimension-mismatch error on the small cameras. Fix:
a second URP renderer **`Instruments_NoSplat`** (no splat feature) assigned to the 5
periscope cameras; VR eyes keep the splat renderer. Consequence: periscope displays
currently show the world *without* splats — acceptable for now, a later refinement.

**I3. New Input System.** The Unity project uses the new Input System exclusively;
`AutoDriveWaypoints.cs` was rewritten to use `Keyboard.current` (with a legacy
fallback). If keys don't respond, check `Project Settings → Player → Active Input
Handling`.

**I4. "Game pauses on Play" is Error Pause + a benign XR startup error**, not a bug.
Meta's XR layer errors if the headset session isn't up when you hit Play; the Console's
**Error Pause** toggle then halts. Start Quest Link first, or turn Error Pause off, or
uncheck "Initialize XR on Startup" for flat-screen iteration.

---

## J. Things deliberately NOT done (and why)

- **True per-splat / hierarchical LOD** (Cesium 3D Tiles style): possible but needs a
  custom streaming renderer — a project on the scale of everything else. Over-engineered
  for a fixed-path tube. The discrete "enable closest 4 sections" approach is enough.
- **Merging a whole video into one giant model**: infeasible on 16 GB (a 300 m video ≈
  11 M splats, ~68 GB training cache). Chunking isn't a workaround, it's the design.
- **Chasing PSNR past ~20.5 dB**: the quality-package ablation (splatfacto-big, absgrad,
  bilateral grid, 2000 px crops, 50k iters) was scoped and **shelved** because it added
  ~45% runtime for marginal gain and the 12 km timeline matters more.
- **Antialiased rasterization mode** in Unity: better looking but exports a PLY
  incompatible with the classic renderers (SuperSplat, aras-p Unity) — not adopted.
