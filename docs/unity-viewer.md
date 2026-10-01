# Standalone Unity viewer

Open **`unity/Track3DGSViewer`** with Unity Hub using **Unity 6000.3.19f1**.
No QuestSBTC, VR3DGS, headset, XR package or private Unity package is required.
This is an editor review project, not a Quest runtime or a game.

It includes URP **17.3.0**, the embedded **wu.yize.gsplat 1.4.0** renderer pinned
to upstream revision `a2bf458d6b16395e6570e9345f9f4408f92684b8`, and the same
local renderer fixes used for the earlier previews. Source, `.meta` GUIDs,
MIT license, third-party notices, upstream hashes and patch history are
included under `Packages/wu.yize.gsplat`. The small viewer's project settings
select Direct3D 12 and global sorting. Use **Tools → Track3DGS → 1. Configure
Viewer Rendering** to recreate the URP assets if needed; restart Unity after
changing graphics API. First open requires public Unity Package Manager access.

## Step 1: route markers

Copy your workspace's `reports/route_preview.json` to
`Assets/Track3DGSData/route_preview.json` inside the viewer (create the folder),
then choose **Tools → Track3DGS → Rebuild Route Preview**. This generates
`Assets/Scenes/Track3DGS_RoutePreview.unity`: route segments, registered camera
positions and directions, source time/station labels, and training boundaries.
It also writes an overview image and build report into `Assets/Track3DGSData`.
The marker positions are camera centres, not a road mesh or collision surface.

## Step 2: model comparison

From the repository's `pipeline` directory:

```powershell
$workspace = (Resolve-Path ..\data\routes\my-route\r001).Path
$run = "$workspace\training\t001"
..\.venv\Scripts\python.exe -m track3dgs.route_preview_models --workspace $workspace --run-root $run --unity-project ..\unity\Track3DGSViewer --regions 0,1
```

The command writes disposable recentered PLY caches, `catalog.json` and the
matching `route_preview.json` into `Assets/Track3DGSData`. Publish one completed
region if only one is available. To include a cleanup result, use the
[`--sky-cleanup` workflow](sky-cleanup.md).

Open **Tools → Track3DGS → Training Model Preview** and create/open its scene.
Choose a chunk and representation, then **Load selected model(s)**:

| Representation | Use |
|---|---|
| Original training result | Inspect raw reconstruction including training context |
| Cleaned, with context | Check initial sky/glitter/needle cleanup |
| Trimmed route core | Review route assembly and seams |
| Sky cleaned core | Compare the optional experimental sky-volume pass |

**Show next neighbour** uses the second slot. Raw/clean contexts intentionally
overlap, so use cores for seam review. The viewer enforces exactly two renderer
slots, clears old references before replacements, and releases their GPU resources
when unloaded. Disabled renderers alone do not guarantee that all imported asset
data has left CPU memory; use **Unload both models** before large GPU jobs.

Move the camera with the route-station slider, a recorded cleanup camera, or
**Use Scene view position for Game camera**. **Before: original core** and
**After: sky cleaned** preserve camera pose and FOV, making changes comparable
in the Game view without Play mode. The recorded cameras retain their original
perspective. For free camera motion use Unity's normal Scene view navigation.

Spark is the default. **Uncompressed reference** supports one model because
the renderer's paired global sort requires compatible Spark assets. Use the
native Python QC for a full-precision pair. A tiny interleaved-depth regression
checks that a Spark pair matches a combined reference in forward and reverse
views; the test also rejects a blank render.

## Coordinates and revisions

The publisher subtracts a regional origin from XYZ before compression, then
places the Unity object at that origin with identity rotation and unit scale.
This reduces half-float quantization far from zero. Covariance and SH coefficients
are unchanged. The importer uses **RUB**, applies the Z reflection once, and
prunes no opacity values. Do not apply the historical `(180,0,0)` rotation.

One data folder holds **one route revision**. A mismatched route hash or stale
marker export is rejected before publication. Use a separate viewer copy or
archive the existing generated data before switching routes. On a changed core,
republishing invalidates its sky comparison rather than reusing an obsolete result.

`--preview-folder` supports another folder below a Unity project's `Assets`,
including the old `Assets/TrackTrainingPreview/Track02` cache destination. The
included viewer reads `Assets/Track3DGSData`; custom integrations must point their
reader at the same configured folder. Regenerate caches on another PC instead
of treating them as the portable model master.

## Self-test without a capture

Use a fresh viewer copy without existing `Assets/Track3DGSData`, then from
`pipeline` run:

```powershell
..\.venv\Scripts\python.exe tests/create_viewer_fixture.py --unity-project ..\unity\Track3DGSViewer
```

This generates two tiny synthetic models, route markers and cleanup variants.
Close that viewer before running the batch check:

```powershell
$editor = 'C:\Program Files\Unity\Hub\Editor\6000.3.19f1\Editor\Unity.exe'
$viewer = (Resolve-Path ..\unity\Track3DGSViewer).Path
& $editor -batchmode -force-d3d12 -projectPath $viewer -executeMethod TrackViewerSetup.VerifyBatch -logFile "$viewer\viewer-check.log"
```

Do **not** add `-nographics`: the check needs a real GPU. Success writes
`Library/standalone-verification.txt` and `TRACK_VIEWER_VERIFIED` in the log.
The check generates/replaces diagnostic scenes in that viewer copy. Generated
data, scenes, screenshots and Unity's Library are excluded from Git.
