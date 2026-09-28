# 01 — Project Overview

## 1. What the project does

The end goal is a **VR driving experience through a real forest track** on Meta Quest 3.

A 360° camera (Insta360, equirectangular, 8K, 30 fps) was mounted on an armed vehicle
that drove a **~12 km forest track**. From that single video pass, the project
reconstructs the world as **3D Gaussian Splatting (3DGS)** models and plays them back
inside an existing Unity battle-tank crew simulator, so the crew sees photorealistic
forest scrolling past as their simulated vehicle moves along the track.

Two hard constraints shape every design decision:

- **Visuals only.** No collision, no physics, no interaction with the reconstructed
  world. It is a backdrop, not a playable level.
- **The viewer rides the vehicle.** The head stays within ~1–2 m of the original
  camera path. This is what makes the problem tractable: 3DGS looks great near the
  capture trajectory and degrades away from it, and we never leave the "tube."

3DGS was chosen over photogrammetry/meshes because it renders foliage — thin leaves,
depth-layered canopy, soft edges — far more convincingly, which is the whole point for
a forest.

## 2. The two sub-projects

This is critical to understand up front: **there are two separate project folders.**

### A. The reconstruction pipeline — `C:\Work\Unity\DSTA\Track3DGS`

This git repository. Python + PowerShell. Turns video into splat models. Everything in
docs 02–04 is about this unless stated otherwise. The `data/` folder here holds all
video, intermediates, and finished models, and is **not** in git (too large).

### B. The Unity VR simulator — `C:\Work\Unity\DSTA\QuestSBTC`

A **pre-existing, separate Unity project**: a full battle-tank crew simulator (driver,
commander, gunner stations, periscope displays, controls, enemy AI, etc. — ~40 C#
scripts that predate this work). Our contribution to it is small and additive:

- The reconstructed splat sections, loaded as `GaussianSplatRenderer` objects.
- Three new scripts for the viewing test: `GSObject.cs`, `GSLODManager.cs`,
  `AutoDriveWaypoints.cs` (doc 05 covers these).
- Render-pipeline configuration to make splats work on Quest (doc 02, §Unity).

The pipeline exports plain `.ply` files; a human imports them into this Unity project
using the **UnityGaussianSplatting** package (aras-p). The two projects are only
coupled by that PLY hand-off.

## 3. Architecture of the pipeline

Four principles, all validated in practice (doc 02 has the reasoning and the rejected
alternatives):

1. **Train big, slice small.** 3DGS trains per ~50 m *cell* (good context, few jobs).
   The trained Gaussians are then diced by position into non-overlapping **10 m tiles**
   for runtime streaming. *In practice so far, each captured section is smaller than a
   cell, so one section = one cell = one model. Slicing into tiles and stitching
   sections into a continuous track is designed and partially built but not yet run on
   Track01 — see doc 05.*

2. **One fixed global metric frame.** Trajectory, sparse cloud, trained splats, and
   tiles all live in a single coordinate system. Training runs with pose normalization
   *disabled* so exported models need no re-alignment. Metric scale comes from vehicle
   speed (no GPS).

3. **Every Gaussian belongs to exactly one tile.** No overlapping tiles. (Overlapping
   independently-trained chunks double-render and shimmer — rejected early, doc 02.)

4. **Mask what isn't the world.** The vehicle hull and the sky are excluded from
   training so no splats are ever created for them. Sky becomes a Unity skybox at
   runtime; the hull is simply absent.

## 4. Runtime model (the LOD / streaming idea)

At 8K the sections are ~0.7–1.1 M splats each (~180–270 MB PLY). On a PC driving a
Quest over Link, VRAM holds dozens of sections but **frame time** is the wall: stereo
72 Hz can sustain roughly **3–4 sections rendering at once** (~2.5–3.7 M splats). So
the runtime keeps many sections *loaded* but enables only the closest ~4 to the
viewer — a cheap visibility toggle, which is what `GSLODManager.cs` does today. This
matches the original "3 active tiles" design, now confirmed by measured section sizes.

## 5. Phases and current status

The project was planned in four phases; the order was deliberately changed early
(Phase 1 first, to have something to show — doc 02).

| Phase | Content | Status |
|-------|---------|--------|
| **1 — Offline reconstruction pipeline** | video → per-section 3DGS models | **DONE & production-proven.** 27 Track01 models delivered |
| **0 — Quest render feasibility** | can Quest 3 render these at 72 Hz? | **Partially answered.** Runs over PC→Quest **Link** (PC renders). True **standalone on-device** rendering not yet tested |
| **2 — Unity runtime streaming** | load/enable tiles as the vehicle moves | **Early.** LOD manager + auto-drive built and unit-verified in-editor; full VR drive test in progress |
| **3 — Scale to 12 km** | batch the whole track; stitch sections | **Not started.** Track01 (~833 m) done; stitching sections into one continuous frame is the key missing capability |

**What "done" means for Phase 1:** a single command reconstructs a section with zero
human intervention, every quality gate self-checks, and a 26-section batch ran 22.5 h
unattended with 26/26 success. See `data/Export/qc_log.csv` for the per-section
quality ledger.

## 6. Hardware

- **Training / reconstruction PC** (the machine everything ran on): Windows 11,
  **NVIDIA RTX 5060 Ti, 16 GB** (Blackwell, compute capability sm_120). This new
  architecture is why the GPU stack is exacting — it needs CUDA 12.8-class toolchains,
  torch cu128, and a JIT-compiled gsplat (doc 04). Also runs the Unity editor and
  Quest Link.
- **NVIDIA DGX Spark** — available, Linux, large unified memory. Intended as a second
  batch worker and the natural home for Linux-only tools (e.g. stella_vslam), but **not
  yet integrated**. All 27 models so far were made on the 5060 Ti alone.
- **Meta Quest 3 / 3S** — the target headset.

## 7. Numbers worth knowing

- Per section (~25–50 m): **~48 min** end-to-end on the 5060 Ti (COLMAP ~11 min,
  training ~33 min, everything else minutes).
- Full 12 km projected: **~15 days** single-machine, **~8 days** with the DGX Spark in
  parallel — *at current per-section SfM cost*. The bigger speed lever (per-video
  SfM / stella_vslam) is discussed in doc 05.
- Quality: masked PSNR **~20.5 dB mean** (19.0–21.8 range) across the 26-section batch.
- Each section ≈ **0.7–1.1 M splats** after pruning; ~180–270 MB per PLY.
