# 05 — Current State & Next Steps

## 1. Exactly where work stopped

The last active work was **the Unity VR viewing test** (LOD streaming + auto-drive).
Immediately before that, the **Track01 reconstruction batch finished**: 26 sections
unattended in 22.5 h, 26/26 success, plus the S1 calibration section = **27 models in
`data/Export/`**. The owner is **hand-polishing each model in SuperSplat**
(`data/Export/Edited/`, ~15 of 27 done at handover).

The very last thing done: fixing the Unity multi-camera render conflict (added the
`Instruments_NoSplat` renderer for the periscope cameras) so the scene renders splats
without per-frame errors. The next action was to actually drive through the scene in
the headset and read the frame rate with 4 sections enabled.

## 2. Done vs pending — the honest status board

**DONE and proven**
- Full reconstruction pipeline, 9 stages, 58 unit tests green.
- One-command section run; unattended, resumable, self-QC'ing batch.
- Track01 fully reconstructed (~833 m, 27 models), quality logged.
- Mount-calibration auto-levelling; union sky masking; 3-tier pruning; global mapper.
- Unity: splats render in-editor and over Quest Link; DX12/Vulkan/URP-feature set up;
  multi-camera conflict resolved.
- LOD manager + waypoint auto-drive scripts written and unit-verified in edit mode.

**PENDING / not yet done**
1. **VR drive test outcome** — the frame-rate reading with 3–4 sections enabled over
   Link on the 5060 Ti. (Envelope estimate: ~3–4 sections at stereo 72 Hz. Unmeasured.)
2. **Standalone on-device rendering** — everything so far renders on the *PC* (Link).
   True Quest-3-standalone (mobile GPU) is the original Phase 0 question and is
   **untested**. It will be far tighter; will likely need harder splat caps / the
   `slice`+`pack` tiling and possibly foveation.
3. **Tiling** — `slice`+`pack` were **not** run on Track01. Deliverables are
   whole-section models, not 10 m tiles. No `manifest.json` exists for Track01.
4. **Stitching sections into one continuous track** — the 27 sections are in **27
   independent coordinate frames** with per-section scale (±speed error). They are not
   yet aligned into one world. This is the biggest missing capability (see §4).
5. **Manual polish** — ~12 of 27 sections not yet hand-cleaned in SuperSplat.
6. **The rest of the 12 km** — only Track01 (~833 m) is captured/cut/reconstructed.
7. **DGX Spark** — not yet used; all work was on the single 5060 Ti.
8. **Periscope splats** — the crew displays currently render the world without splats
   (doc 02 §I2).

## 3. The Unity runtime scripts (as they stand)

Three scripts in `QuestSBTC/Assets/Scripts/`, all written this session:

**`GSObject.cs`** — a marker. Add it next to any `GaussianSplatRenderer`; the object
becomes LOD-managed. Optional `centerOverride` transform if the object origin isn't its
visual centre (a section's origin is its trajectory *start*, so this can help). Exposes
`RendererEnabled` (toggles the splat renderer's `enabled`).

**`GSLODManager.cs`** — one instance in the scene. Every `updateInterval` (0.25 s) it
sorts all registered `GSObject`s by distance from `trackingTarget` (a Transform — set to
the ControlRoom; falls back to `Camera.main`) and enables only the closest `maxVisible`
(default 4), disabling the rest. This is the "load closest 4 of 12" behaviour. Distance
is from object origins — watch for a section ranking oddly if its origin is at one end.

**`AutoDriveWaypoints.cs`** — on the ControlRoom. `waypointsParent`'s children (in
order) are the path. **T** = teleport to waypoint 0 facing waypoint 1 (horizontal
only). **F** = start / pause / resume. Constant `speed` (2.8 m/s ≈ 10 km/h),
exponentially-smoothed position (rounded corners) and **yaw-only** rotation (never
tilts). Pauses at the last waypoint. Uses the new Input System (`Keyboard.current`).

**To run the test**: place waypoints along the road at splat ground height under a
`Waypoints` object; connect Quest Link; Play; press **T** then **F**; watch sections
enable/disable in the Hierarchy and read the frame rate. Known caveat: a section
re-enabling re-uploads GPU buffers (a brief hitch) — acceptable for the test; a
production manager would pre-warm instead of hard-toggling.

## 4. Recommended roadmap (in priority order)

**(a) Finish the VR drive test and record the numbers.** How many sections hold 72 Hz
over Link? This calibrates everything downstream (how many tiles can be active, whether
standalone is viable). Cheap, high-information.

**(b) Decide the stitching / frame-unification strategy — the key unlock.** The 27
independent frames must become one continuous world before a real "drive the whole
track" experience exists. Two routes (discussed in-session):
- **Stitch after the fact**: register each section to its neighbour via the intentional
  ~2 s overlaps (Sim3 per section — the overlap frames saw identical scenery, so it's
  well-posed). Reconciles both orientation and the per-section scale drift.
- **One SfM per video** (the spec's original architecture): run COLMAP/stella_vslam once
  per video for a single shared frame + scale, then auto-carve 50 m cells from the one
  trajectory. Tiles then align by construction. Needs affordable per-video SfM
  (subsampled-view GLOMAP, or stella_vslam on the DGX Spark). **This is the better
  long-term answer and the biggest single simplifier for Unity assembly.**
Either way, once unified, run `slice`+`pack` to get tiles + `manifest.json`, and build a
manifest-driven tile-streaming manager in Unity (replacing the simple distance toggle).

**(c) Standalone Quest test.** Only meaningful after (a). If PC-Link is comfortable but
standalone isn't, the fixed-path design still allows a tethered/streamed deployment.

**(d) Scale-out.** Bring in the DGX Spark as a second batch worker (training is
embarrassingly parallel across sections). Capture + cut the remaining ~11 km. For the
full 12 km, seriously evaluate stella_vslam to replace per-section COLMAP (the ~11 min
mapper × hundreds of sections is the dominant cost).

## 5. Faster / cheaper ideas already vetted (don't re-derive)

- **Automatic uniform cutting** (fixed 10 s sections with 2 s overlap, no manual
  timetable) is nearly free to add for future tracks — the owner's hand-curation mostly
  encoded exactly that. The QC gates catch bad stretches automatically.
- **Global mapper** already gave the 3× SfM speedup; the next lever is per-video SfM /
  stella_vslam, not more COLMAP tuning.
- **Discrete per-tile LOD** (2–3 quality levels per tile via the existing pruning, chosen
  by distance ring) is the pragmatic way to stretch the visible corridor if frame time
  gets tight — much cheaper than a custom hierarchical renderer (doc 02 §J).

## 6. If something breaks — first moves

- Pipeline stage fails: check it isn't a Windows gotcha (doc 02 §H3) — UTF-8, plyfile
  mmap, torch unpickling, or the **nerfstudio pin_memory patch** having been lost after
  a reinstall.
- Orientation looks wrong on new footage: the **mount key** is stale for that
  footage/mount — re-measure it (one SuperSplat levelling, doc 02 §D2).
- Splats invisible in Unity: the three settings (DX12 / Vulkan / `GaussianSplatURPFeature`
  on the renderer) — doc 02 §I1.
- Batch stalled: it's resumable — just rerun `run_track01_batch.ps1`; it skips finished
  sections and rebuilds the interrupted one.
- Trust the numbers: `data/Export/qc_log.csv` tells you which sections are weakest
  (lowest `psnr_db`, or any `guard_ok=False`) — polish or re-run those first.

## 7. One-paragraph mental model to keep

*Body-locked 360° video → per-section pinhole crops → COLMAP global SfM in one metric
frame → auto-level against a once-measured mount key → Splatfacto trains a masked model
with pose-normalization off so it stays in that frame → 3 pruning tiers strip sky and
artifacts → export a whole-section PLY. Twenty-seven of these exist for Track01. The
unbuilt future is joining them into one continuous, tiled world and streaming ~4 tiles
at a time in the Quest tank sim.* Everything else is detail in docs 02–04.
