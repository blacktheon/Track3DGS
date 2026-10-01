# View-guided automatic sky cleanup

This optional Step 2 pass removes residual sky curtains and elongated splats
using the calibrated source cameras. It produces a separate experimental PLY;
it never overwrites the original training, clean or core model. It is a quality
cleanup, not the large percentage reduction performed by VR3DGS Stage 1.

## Algorithm

1. Select route stations and yaw views, or explicit registered image names.
   Project each panorama's **sky-only** mask using the original pinhole FOV.
   Keep components connected to the image's top edge. Black vehicle pixels and
   isolated dark foreground pixels are not removal targets.
2. Erode the safe sky boundary by eight pixels at the original 1600-pixel crop
   resolution. Project Gaussian centres using the registered camera pose.
3. Measure each Gaussian's *visible alpha contribution* inside safe sky and
   valid foreground: differentiating a rendered scalar color channel with
   respect to each Gaussian's independent color gives its accumulated alpha
   contribution. Re-render after removal to expose previously hidden layers.
4. The conservative `consensus` mode requires evidence from two distinct
   panorama positions. Overlapping yaws at one position count once. Two centre
   votes are sufficient; footprint-only candidates additionally need at least
   5% of their measured contribution in safe sky. A view vote requires more than
   0.05 accumulated alpha-pixels. Stop when no new rows are selected, or after
   eight rounds; report whether convergence occurred.
5. Copy retained rows and their source identities exactly, including all SH3
   coefficients, covariance parameters and opacity. Render before/after views
   and inspect foreground damage as well as sky improvement.

The earlier `single` mode intentionally operates on one edit camera. It seeds
removal using centres and sampled projected ellipses: 16 area-spaced radial
rings × 48 angles to 3 sigma, retaining samples with alpha ≥1/255. The same
visible-alpha refinement catches large or thin spikes missed by sampling.
This mode is more aggressive, has no multi-position foreground protection,
and is an approximation to Gaussian support, not exact volume clipping.

## Run

From `pipeline`, using the environments in [setup](setup.md):

```powershell
$utility = (Resolve-Path ..\.venv\Scripts\python.exe).Path
$gpu = (Resolve-Path ..\.venv-train\Scripts\python.exe).Path
$workspace = (Resolve-Path ..\data\routes\my-route\r001).Path
$run = "$workspace\training\t001"
$cleanup = "$run\experiments\sky-cell000-v001"

# Copy/edit the example's stations and yaws for your region first.
& $utility -m track3dgs.sky_volume prepare --workspace $workspace --config configs/sky-track02-model1.json --output $cleanup
.\cuda_env.bat $gpu -m track3dgs.sky_volume carve --workspace $workspace --source "$run\models\cell_000\core.ply" --output $cleanup
.\cuda_env.bat $gpu -m track3dgs.sky_volume evaluate --workspace $workspace --output $cleanup

# Add the experimental result and recorded cameras to the standalone viewer.
& $utility -m track3dgs.route_preview_models --workspace $workspace --run-root $run --unity-project ..\unity\Track3DGSViewer --regions 0 --sky-cleanup $cleanup
```

Use a **new output directory** for each parameter change. `prepare` runs in the
utility environment (it needs py360convert); `carve` and `evaluate` need CUDA.
Supply native route-frame PLYs, never the recentered Unity caches. A raw PLY
without provenance needs `--source-index` matching its region index.

The JSON selection supports `edit`, `validation` and `check` groups. Each group
has either `split`, `stations`, `yaws`, or a `names` array. `validation` requires
held-out panorama identities disjoint from every edit frame. A frame used for
editing is no longer independent evidence, even if originally held out from
training. Sparse selections that collapse several requested stations onto the
same position fail rather than inventing additional votes.

`selection.json` records source route/camera hashes, generated-mask hashes,
intrinsics, parameters and chosen cameras. `processing.json`, `classification.npz`
and `progress.json` record the decision and iteration history. `models/sky.ply`
and its `.provenance.bin` are the result. `index.html`, `metrics.json` and
`renders/` provide review evidence; foreground low-alpha and PSNR are diagnostics,
not automatic acceptance thresholds.

## Reproduce the two Track02 experiments

With the original local `r002/training/t001` data:

| Experiment | Configuration | Carve input | Result |
|---|---|---|---:|
| Model 1, five positions × four yaws | `sky-track02-model1.json` | `cell_000/core.ply` | 378,428 → **373,552** |
| Model 2, one edit camera | `sky-track02-model2.json` | `cell_001/splat.ply`, `--source-index 1` | 612,278 → **608,135** |

For Model 2, transfer the raw deletion to its existing core by provenance:

```powershell
& $utility -m track3dgs.sky_volume subset-core --workspace $workspace --output $cleanup --core "$run\models\cell_001\core.ply"
```

This produces `models/sky_core.ply`: **437,360 → 434,740**. The command checks
that the core's attributes are identical to the original source rows before
transferring deletion. The publisher automatically chooses this core derivative.

The standalone extraction reproduced both experiments byte-for-byte, including
their provenance. Model 1's 20 independent validation views showed a 40.0%
reduction in mean sky alpha, with foreground low-alpha increasing by 0.69
percentage points. Some vegetation thins, uncertain spikes remain, and road
reconstruction defects are not repaired by deleting sky splats. The attractive
single-camera Model 2 example is an **edit-reference** result, not proof of
equivalent quality around the entire region.

After reviewing the cleanup, keep its PLY/provenance/report together for manual
preprocessing. The existing `package_export` command still exports the baseline
route cores; it does **not** silently promote experimental sky results into the
accepted reconstruction package.
