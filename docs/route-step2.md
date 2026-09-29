# Step 2: regional training and route assembly

This opt-in workflow reuses the tested Splatfacto backend while keeping every region in the reviewed Step 1 coordinate frame. It does not modify legacy section outputs or manually edited PLYs. Training, visual seam acceptance, Stage1 reduction and application performance are separate milestones.

## Requirements

- Windows; utility environment with the pipeline dependencies, and the existing GPU environment: Python 3.11, torch 2.7.1+cu128, Nerfstudio 1.1.5, gsplat 1.4.0, CUDA 12.8 and MSVC v143. The route-only image-cache adapter deliberately rejects other Nerfstudio versions.
- A connected, reviewed Step 1 revision containing `route.json`, `regions.json`, `cameras.jsonl`, views, masks and region COLMAP subsets.
- A new training revision directory and `route_review.json` recording the exact SHA-256 of the reviewed `route.json`, the review decision and calibration limitations. A review record authorizes training, not visual acceptance of the eventual model.
- GPU memory shared with the Unity editor must be observed during the pilot. Only one training/QC subprocess runs at a time. The Windows GPU lease prevents a second route coordinator from starting concurrently.

## Run sequentially

From `pipeline`, with paths adjusted to your checkout:

```powershell
$utility = 'C:\Work\Unity\DSTA\Track3DGS\.venv\Scripts\python.exe'
$gpu = 'C:\Work\Unity\DSTA\Track3DGS\.venv-train\Scripts\python.exe'
$workspace = 'C:\Work\Unity\DSTA\Track3DGS\data\routes\track02\r002'
$run = "$workspace\training\t001"
& $utility -u -m track3dgs.route_step2 --workspace $workspace --run-root $run --training-python $gpu --unity-project 'C:\Work\Unity\DSTA\QuestSBTC' --regions 0,1,2,3,4,5 --iterations 30000
```

The coordinator trains and exports one region, cleans it, assigns its route core, renders held-out views, checks its seam with the preceding completed region, then publishes its Unity cache. It never trains all regions concurrently. `step2_state.json` records the active region/stage, completed regions from that invocation and any failure. Full logs are in `reports`. Restart with the same inputs to reuse verified completed training; checkpoints resume interrupted training. Changed training settings require a new revision. QC may rerun on resume.

Use `route_training` for a bounded GPU probe without publishing models, for example `--regions 0 --iterations 200` into a separate `training/smoke001` directory. A smoke model is not a completed training result. `route_assembly`, `route_qc` and `route_preview_models` also expose individual command-line entry points; use `--help` for their arguments.

## What each chunk contains

`models/cell_NNN/` keeps:

| File | Meaning |
|---|---|
| `splat.ply` | Original native SH3 export, including training context |
| `model.json` | Export hash/count, native transform, training identity and sparse alignment result |
| `clean.ply` | Sky/glitter/needle cleanup; context remains |
| `core.ply` | Cleaned rows owned by this route core |
| `*.provenance.bin` | Packed little-endian uint32 source index + uint64 original row, 12 bytes per retained row |
| `processing.json` | Counts, cleanup settings and ownership convention |
| `unity_cache.json` | Disposable preview conversion and position-quantization evidence |

Original masters are never rotated, stripped to SH0, or overwritten by cleanup. The route trainer disables Nerfstudio's implicit COLMAP world-axis rotation as well as centering, orientation fitting, scale normalization and camera optimization. The recorded dataparser transform must be identity. This preserves geometry, covariance and directional colour in the same right-handed RUB frame. The legacy `NS_EXPORT_FIX` is not applied to route exports.

The first Track02 pass uses existing globally registered 2 Hz keyframes, with 1600-pixel crops. Every crop of a held-out panorama stays outside the training set. The bounded image cache holds at most 32 decoded training images and eight evaluation images; it changes loading, not pixel values or the training loss. Dense image registration is a later refinement if the first models reveal undersampled regions.

Cleanup reuses the mask-projection and needle tests. The canopy-height condition is measured above the local route, preserving the meaning of the test on slopes. Only training-frame sky masks inform cleanup. It does not use the legacy independent section end cuts.

## Ownership and visual evidence

Centres project onto the closest **continuous route segment**, rather than onto a sampled camera. Core intervals are half-open except the last inclusive endpoint. Training context overlaps, while exported core domains do not. Gaussian footprints are not clipped. Nearby nonadjacent route branches are flagged for an explicit ownership decision.

`reports/qc/cell_NNN/index.html` compares held-out photographs, original/clean renders and alpha. `boundary_NNN/index.html` compares each neighboring cleaned reference and the globally sorted pair of trimmed cores. Photographs are shown with the same sky/vehicle mask. PSNR and low-alpha coverage are diagnostic evidence; there is no invented numeric threshold that declares a seam accepted. Inspect colour, road continuity, vegetation silhouettes, holes and duplicate structures. At most two Gaussian models are resident in each QC render.

If a seam fails, preserve the result and use the evidence to choose a wider context, denser image registration, revised boundary, or shared repair region. Automated context-expansion retries and synthetic lateral sweeps are not implemented in this first pass. No pipeline status currently grants human visual acceptance automatically.

## Unity review in QuestSBTC

The selected renderer is the pinned `wu.yize.gsplat` 1.4.0 package already used by VR3DGS, including its recorded local patches. The new scene is `Assets/Scenes/Track02_TrainingPreview.unity`; open **Tools → Track3DGS → Training Model Preview**.

Select a completed chunk, original/clean/core representation, and optionally its next neighbor. Click **Load selected model(s)**. For seam review use **Trimmed route core**. Raw and cleaned context views intentionally overlap. Move the route-station slider or use a Scene view position for the Game camera. The preview is for editor inspection, not the game's driving/VR controls.

Exactly two renderer slots exist. Switching clears both slots and releases their GPU resources before loading replacements. The original Step 1 marker scene remains separate. Global sorting is enabled for adjacent pairs. Spark compression is the efficient default; **Uncompressed reference** enables a precision comparison. Each cache subtracts a regional origin before compression and places the object back at that origin in Unity, reducing float16 position error far from world zero. Only translation is baked, so SH coefficients and covariance remain unchanged. The importer applies RUB-to-Unity conversion once; the object translation reflects package Z once. The source portable PLY remains untouched.

Generated cache PLYs are local data, not files to commit to Git. Recreate them with `route_preview_models` when moving to another checkout, then load the models through the preview window to refresh scene references.

## Portable draft handoff

After the intended regions are processed:

```powershell
& $utility -m track3dgs.package_export --workspace $workspace --run-root $run --output "$workspace\packages\training-t001" --revision training-t001
& $utility -m track3dgs.package_contract "$workspace\packages\training-t001"
```

Export copies core PLYs, source-row provenance, shared route/camera metadata and available QC evidence into a staging directory. It verifies hashes, lengths, transforms, SH layout, covariance bounds and identities before publishing a new immutable directory. An existing output directory is never overwritten. Original full training models remain in the training revision and are identified by hash in the source registry.

The output is always `validation.state: draft`. The package contract and a native PLY are available for downstream adapter work; this command does not claim a complete generic VR3DGS importer, LOD generation or a measured Quest 3 runtime budget.
