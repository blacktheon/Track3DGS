# Step 2: regional training and route assembly

This opt-in workflow reuses the tested Splatfacto backend while keeping every region in the reviewed Step 1 coordinate frame. It does not modify legacy section outputs or manually edited PLYs. Training, visual seam acceptance, Stage1 reduction and application performance are separate milestones.

## Requirements

- Windows; utility environment with the pipeline dependencies, and the existing GPU environment: Python 3.11, torch 2.7.1+cu128, Nerfstudio 1.1.5, gsplat 1.4.0, CUDA 12.8 and MSVC v143. The route-only image-cache adapter deliberately rejects other Nerfstudio versions.
- A connected, reviewed Step 1 revision containing `route.json`, `regions.json`, `cameras.jsonl`, views, masks and region COLMAP subsets.
- A new training revision directory and `route_review.json` recording the exact SHA-256 of the reviewed `route.json`, the review decision and calibration limitations. A review record authorizes training, not visual acceptance of the eventual model.
- GPU memory shared with the Unity editor must be observed during the pilot. Only one training/QC subprocess runs at a time. The Windows GPU lease prevents a second route coordinator from starting concurrently.

## Run sequentially

From `pipeline`, after installing the [utility/GPU environments](setup.md):

```powershell
$utility = (Resolve-Path ..\.venv\Scripts\python.exe).Path
$gpu = (Resolve-Path ..\.venv-train\Scripts\python.exe).Path
$workspace = (Resolve-Path ..\data\routes\my-route\r001).Path
$run = "$workspace\training\t001"
New-Item -ItemType Directory -Force $run | Out-Null
```

After inspecting and accepting your Step 1 route, record that decision for its
exact hash. This records your review; it does not perform the review for you.

```powershell
@{
    schema_version = 1
    route_sha256 = (Get-FileHash "$workspace\route.json" -Algorithm SHA256).Hash.ToLowerInvariant()
    accepted_for = 'Step 2 first training pass'
    human_review = 'I inspected the route, camera directions, coverage and scale assumptions'
    scale = 'Approximate; replace with the actual basis for this capture'
    seam_acceptance = 'pending'
} | ConvertTo-Json | Set-Content "$run\route_review.json" -Encoding utf8

# Start with one region. No Unity project is required.
& $utility -u -m track3dgs.route_step2 --workspace $workspace --run-root $run --training-python $gpu --regions 0 --iterations 30000
# After reviewing the pilot, continue the region indices in your regions.json.
# Add --unity-project ..\unity\Track3DGSViewer to publish review caches along the way.
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

If a seam fails, preserve the result and use the evidence to choose a wider context, denser image registration, revised boundary, or shared repair region. Automated context-expansion retries are not implemented. `route_lateral_qc --workspace <workspace> --run-root <run>` now provides the synthetic lateral seam sweep; its default forward/backward yaws (180/0 degrees) reflect the Track02 mounting and can be changed. The [Track02 t001 results](track02-t001-results.md) include a separate one-run synthetic lateral diagnostic and its limitations. No pipeline status currently grants human visual acceptance automatically.

## Standalone Unity review

Use the included `unity/Track3DGSViewer` project; QuestSBTC is not required.
The [viewer guide](unity-viewer.md) covers installation, route-marker scenes,
raw/clean/core/sky comparison, global pair sorting, recorded viewpoints and
resource limits. `route_preview_models` recreates caches independently of
training. It defaults to `Assets/Track3DGSData`, copies the route-marker export,
and rejects different route revisions in an existing catalog.

The optional [sky-volume cleanup](sky-cleanup.md) preserves full retained
attributes and provenance. Its experimental results are separately reviewable;
the baseline package export below continues to use `models/cell_NNN/core.ply`.

## Portable draft handoff

After the intended regions are processed:

```powershell
& $utility -m track3dgs.package_export --workspace $workspace --run-root $run --output "$workspace\packages\training-t001" --revision training-t001
& $utility -m track3dgs.package_contract "$workspace\packages\training-t001"
```

Export copies core PLYs, source-row provenance, shared route/camera metadata and available QC evidence into a staging directory. It verifies hashes, lengths, transforms, SH layout, covariance bounds and identities before publishing a new immutable directory. An existing output directory is never overwritten. Original full training models remain in the training revision and are identified by hash in the source registry.

The output is always `validation.state: draft`. The package contract and a native PLY are available for downstream adapter work; this command does not claim a complete generic VR3DGS importer, LOD generation or a measured Quest 3 runtime budget.
