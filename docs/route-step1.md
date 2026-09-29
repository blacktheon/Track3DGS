# Continuous route reconstruction — Step 1

This opt-in workflow reconstructs one coordinate frame from a continuous 360° video, then proposes overlapping regions for later Gaussian training. It does not train Gaussians, export cleaned PLYs, or create runtime LODs. The existing section workflow remains available.

## Run

From `pipeline`, using the existing utility environment:

```powershell
..\.venv\Scripts\python.exe -m track3dgs.route --config configs/track02-route.json --dry-run
..\.venv\Scripts\python.exe -u -m track3dgs.route --config configs/track02-route.json
```

The checked-in Track02 configuration contains workstation-specific paths. Copy it for another video and set the source video, workspace, vehicle mask, sky prior and existing training Python executable. Paths are resolved relative to the configuration file. Use a **new revision directory** when changing inputs or reconstruction settings. The coordinator resumes completed stages but never resets a workspace or overwrites a different manual vehicle mask.

`--through ingest`, `views`, `reconstruct`, or `regions` stops after the selected stage. The default is `regions`. A failed coverage gate writes diagnostic route images but withholds region planning and exits with an error.

## Track02 first-pass profile

| Setting | Value |
|---|---|
| Input | 210.033-second, 7680 × 3840, 30 fps video |
| Sharpness | Existing best-of-three selection; original presentation timestamps retained |
| Route keyframes | 2 Hz, including the last selected frame |
| Pinhole views | Eight yaws, 1600 × 1600, 100° field of view |
| Masks | Existing fixed vehicle mask plus existing union sky masking |
| Reconstruction | Existing COLMAP global mapper; overlap 48; fixed projected intrinsics |
| Scale | One global scale from an **assumed** 10 km/h average speed |
| Orientation | One mean camera-up alignment and initial travel heading; no measured gravity or north |
| Initial training regions | 100 nominal-metre cores, 20 nominal-metre context on either side |
| Capture margins | 20 nominal metres at both ends; still usable as training observations |
| Photographic holdouts | Every tenth selected route panorama and all its crops |

The 2 Hz solve establishes the common route frame at manageable cost. Before dense training, register additional best-of-three frames into this accepted frame, maintain the existing held-out source-frame identities, and validate regional coverage. Do not run independent region mapping or normalization.

Video-only reconstruction can drift. The nominal speed establishes total scale, **not** each frame's position. Position, curves and height variation come from image correspondences. Absolute grade remains uncertain, and camera positions are not ground contact points or a driveable road mesh.

## Outputs

All generated data goes under the configured workspace, normally `data/routes/track02/r001/` (Git-ignored).

| Output | Purpose |
|---|---|
| `route_config.resolved.json`, `state/` | Input identity, settings and resumable stage completion |
| `frames_meta.jsonl` | Source decoded index, original PTS, SHA-256 frame identity, split and sharpness |
| `frames/`, `sky_masks/`, `views/`, `views_masks/` | Selected panoramas and masked pinhole observations |
| `track/colmap_work/` | Source database and untouched raw SfM models |
| `track/colmap/` | Globally transformed sparse evidence for later training |
| `route.json` | One route frame, source transform, scale basis, timestamps, arc length and rig poses |
| `cameras.jsonl` | Registered pinhole camera-to-package poses and stable source-frame splits |
| `regions.json` | Proposed core/context intervals, camera memberships and nearby-branch warnings |
| `cells/cell_*/train`, `held_out` | Sparse COLMAP subsets preserving the global coordinates |
| `reports/index.html`, `route_overview.png` | Human-readable route and region review |
| `reports/training_regions.csv` | Video-context intervals for each proposed region |
| `reports/route_preview.json` | Explicitly converted Unity camera markers |

Core intervals are half-open; only the final core includes its endpoint. Training contexts intentionally overlap. No video is physically cut in Step 1: each region references original frames. Use the context time intervals if separate clips are needed, without remapping or changing source identities.

Coverage currently requires one reported reconstruction component, at least 95% of selected rig frames, and no registration gap above two seconds. This gate does not prove a route is geometrically correct. Inspect the shape, camera consistency and regional boundaries before accepting it. Nearby nonadjacent route branches are flagged for later seam ownership review; no splats are removed here.

## QuestSBTC visual preview

Copy `reports/route_preview.json` to `Assets/TrackRoutePreview/Track02/route_preview.json` in QuestSBTC and use **Tools → Track3DGS → Rebuild Track02 Route Preview**. The editor utility creates `Assets/Scenes/Track02_RoutePreview.unity`, with visible camera markers, camera-forward arrows, coloured route segments, time/distance labels, capture endpoints and training-core boundaries. Grey marks show capture margins. A top-down overview camera renders the scene without a headset.

The exporter converts the package's right-handed RUB frame into Unity's left-handed frame by reflecting Z once. OpenCV camera down is converted to camera up separately. Do not apply another Z flip when importing the preview.

The preview scene is separate from gameplay scenes and is not added to build settings. It is diagnostic geometry, not a trained track, collision mesh or tank controller.

## Checks

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Tests cover timestamp identity, long FFmpeg selections, immutable masks/configuration, curved/sloped/reversing paths, global scale, disconnected or missing coverage, training-context overlap, grouped holdouts, and Unity coordinate conversion. Real-video route acceptance additionally needs inspection of generated reports and the Unity scene.
