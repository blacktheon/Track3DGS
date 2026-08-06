# Track3DGS — 360° Track Video → Chunked 3DGS → Unity Quest Streaming

**Status:** DRAFT — under review
**Date:** 2026-08-06

## 1. Goal

Reconstruct a 12 km forest track (captured as 8K 30 fps equirectangular 360° video from a
camera mounted on an armed vehicle) as 3D Gaussian Splatting models, divided into ~10 m
runtime tiles. A Unity app on **Meta Quest 3/3S (standalone)** simulates driving the track,
keeping only ~3 tiles loaded at a time and streaming tiles in/out as the vehicle moves.

**Constraints & simplifiers**

- Visual quality only — no collision, no physics interaction with the environment.
- Viewer is locked to the vehicle: head stays within ~1–2 m of the original capture
  trajectory. Reconstruction only needs to look good from this "tube."
- Source footage: 7680×3840, 30 fps, mp4 (already captured). GPS not required
  (see §5 Scale & drift).
- Training hardware: RTX 5060 Ti workstation and/or DGX Spark; both can run cell
  training jobs in parallel during scale-up.
- First milestone: a 30–40 m test section, end to end, before scaling to 12 km.

## 2. Core architectural decision (Approach B)

**Train big, slice small; partition space instead of overlapping it.**

- **Training unit = cell (~50 m core + ~10 m shared padding each side).** Trained in a
  *fixed global coordinate frame* (pose normalization disabled). ~240 cells for 12 km
  instead of ~1,200 chunk trainings.
- **Runtime unit = tile (10 m).** Produced *after* training by bucketing each cell's
  Gaussians by position along the track. Every Gaussian belongs to exactly one tile —
  tiles never overlap, so there is **no re-alignment step and no cross-fade blending**.
  Tile boundaries within a cell are seamless by construction (co-trained Gaussians).
  Cell-to-cell boundaries occur every ~50 m and are softened because both cells train
  on identical shared padding footage; residual seams are fixed by local fine-tuning,
  not by runtime blending.

Rejected alternative (original draft): independent 10 m chunk training + SE(3)
re-alignment + runtime opacity cross-fade. Rejected because overlapping Gaussian sets
interleave in the depth sort, double up in opacity, and never agree exactly on geometry
— visible artifacts exactly where the viewer looks.

## 3. Project phases

Execution order: **Phase 1 → Phase 0 → Phase 2 → Phase 3.** Phase 1 comes first to
produce an investor-ready visual as early as possible; the Quest feasibility spike
(Phase 0) follows before any Unity runtime work begins.

### Phase 1 — Offline reconstruction pipeline (proven on 30–40 m test section) — FIRST

Seven scriptable stages (below, §4). No manual touch-ups anywhere — the same scripts
must later run ~240× unattended.

**Investor demo deliverable:** an interactive desktop fly-through of the reconstructed
test section (SuperSplat or nerfstudio viewer) — full splat counts, no mobile
compromises, deliberately not gated on Quest work. No rendered video is produced;
if a shareable mp4 is ever needed, it is a small add-on (camera path + `ns-render`),
not a scheduled deliverable.

**Execution as sub-projects:** Phase 1 is built as five sequential sub-projects, each
with its own mini-spec/plan and a visual smoke test before the next starts:
1. **Video Cutter** — Gradio UI, lossless keyframe-snapped A→B stream-copy cut
   (section selection feeding `extract`).
2. **Frames & Views** — `extract` + `views` as batch CLIs; smoke test by inspecting
   output frames/crops/masks.
3. **Trajectory** — `track` (stella_vslam + metric scale); smoke test via trajectory
   plot + sparse points.
4. **First Splat** — `cells` + `train` on one cell in the global frame; smoke test by
   flying through the section in SuperSplat.
5. **Tiler** — `slice` + `pack`; smoke test by toggling 10 m tiles in SuperSplat and
   inspecting boundaries; produces the manifest Unity consumes.

**Provisional splat budget:** since Phase 0 runs later, training uses a provisional cap
(assume ~1.2 M visible splats on device → ~400 K per 10 m tile at 3 active tiles;
train cells at a generous cap, e.g. 2–3 M). If Phase 0 later measures a lower budget,
the fix is re-running `slice`/`pack` with harder pruning — cheap — with cell retraining
at a lower cap only as a last resort.

### Phase 0 — Quest 3 rendering feasibility spike — after Phase 1, before Phase 2

Stand up Unity (URP + OpenXR) on Quest 3 rendering a 3DGS scene — now able to use a
real tile from Phase 1 as the test asset.

- Candidate renderers: `UnityGaussianSplatting` (aras-p) and mobile-capable forks;
  evaluated **on device**, not in editor.
- Deliverables: max splat count at stable 72 Hz, GPU/CPU memory per splat, tile-sized
  asset load time, chosen runtime asset format.
- These numbers set the final per-tile splat cap (applied via `slice`/`pack` pruning)
  and the tile manager's resident tile count.
- Feasibility reference: Niantic Scaniverse renders 3DGS natively on Quest 3.

### Phase 2 — Unity runtime tile streaming

Tile manager + vehicle rig consuming Phase 1 output (§6), using the renderer and
budgets chosen in Phase 0.

### Phase 3 — Scale-up to 12 km + polish

Batch orchestration (resumable job queue over both GPUs), storage compression audit,
per-cell automated QC, ODGS side-experiment (§8).

## 4. Offline pipeline — seven stages

Each stage is a standalone Python CLI reading/writing a shared on-disk project layout;
any stage can be re-run for any single cell.

| # | Stage | In → Out | Key points |
|---|-------|----------|------------|
| 1 | `extract` | mp4 → frames + metadata | FFmpeg extraction; sharpness filter (variance of Laplacian) drops motion-blurred frames; static **vehicle-body mask** authored once in equirect space (vehicle roof/mounts visible near nadir in every frame must be excluded downstream) |
| 2 | `views` | frames → perspective crops | `py360convert`; 4–6 yaw directions (±90° sides, ±45° obliques, optional ±135° rear-obliques), ~100° FOV, ~1600 px, one shared pinhole intrinsic; mask carried through per view |
| 3 | `track` | equirect video → global poses + sparse cloud | `stella_vslam` equirectangular tracking; metric scale fixed from a known distance or average speed; per-view poses derived from fixed rig yaw offsets; exported as COLMAP-format model. Fallback: overlapping VSLAM segments stitched by rigid alignment on shared frames |
| 4 | `cells` | trajectory → cell definitions | Arc-length parametrization; 50 m core + 10 m padding per side; global manifest JSON; everything stays in one global frame |
| 5 | `train` | cell views + poses → cell PLY (global coords) | `gsplat` (Splatfacto or plain gsplat trainer); **pose normalization disabled**; MCMC densification with a generous provisional cap (§3 Phase 1); final Quest budget applied later in `slice`/`pack` |
| 6 | `slice` | cell PLY → 10 m tile PLYs | Bucket Gaussians by arc-length; padding Gaussians owned by the cell whose core contains them; prune Gaussians with ~zero contribution from the vehicle-path tube (expect 30–50 % reduction) |
| 7 | `pack` | tile PLYs → runtime assets + manifest | In Phase 1: emit tile PLYs + manifest (tile id → arc-length range, file, splat count, bounds) — sufficient for the desktop investor demo. After Phase 0: add conversion to the chosen Quest runtime format (raw PLY at 12 km ≈ 50–100 GB; compressed formats give ~10–20×) |

Data flow is strictly forward; re-running a cell touches only that cell's files.

## 5. Scale & drift (why GPS is optional)

Monocular VSLAM has no metric scale → recover a single global scale factor from a known
distance, known track length, or speed logs. Over 12 km without loop closures the
trajectory will drift, but the chunk boundaries, track shape, height profile, and the
simulated vehicle path all derive from the *same* trajectory — the world and the path
bend together, so the rider never perceives it. GPS becomes necessary only for
georeferencing against external/real-world coordinates. If coarse GPS exists, use it as
a soft prior to bound drift; otherwise process in overlapping segments.

## 6. Unity runtime design

- **TileManager (C#):** maps vehicle arc-length position → active window
  {previous, current, next} (possibly 5 while budget allows). Prefetches the next tile
  asynchronously ~1 tile ahead; deactivates + releases tiles behind. Ring-buffer of
  renderer instances; no GPU blending state. Optional ~0.2 s whole-tile fade to soften
  pop-in at the far boundary (a tile 10 m ahead enters view small and distant — pop is
  minor).
- **Vehicle rig:** follows a spline built from the same global trajectory (position +
  heading + height). Speed control independent of capture speed.
- **Renderer:** whichever plugin/fork Phase 0 selects; one renderer instance per
  resident tile.
- **Memory guard:** hard cap on resident splat count; manifest carries per-tile counts
  so the manager can refuse/delay loads rather than OOM.
- **No collision, no physics** — tiles are render-only.

## 7. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| No Unity renderer hits 72 Hz on Quest 3 | Phase 0 runs before any Unity runtime work; fallbacks: harder pruning via `slice`/`pack` re-run (no retraining), reduced render scale + foveation, or renegotiate target (PC-VR). Investor demo is desktop-based and unaffected |
| Phase 0 measures a lower splat budget than the provisional cap used in Phase 1 training | `slice`/`pack` re-run with harder pruning (cheap); cell retraining at a lower cap only as last resort |
| VSLAM tracking loss (blur, vibration, repetitive forest) | Sharp-frame selection; overlapping segment stitching; COLMAP/OpenMVG fallback per segment |
| Motion blur / rolling shutter degrade training | Sharpness filtering in `extract`; if insufficient, deblur pass or recapture at higher shutter speed |
| Exposure changes along track cause flicker between frames | Prefer locked exposure at capture; otherwise per-image appearance embeddings — used cautiously (can cause view-dependent flicker), or exposure equalization pre-pass |
| Moving objects (other vehicles, personnel) bake in as ghosts | Segmentation masking pass in `views` (flag as needed per footage review) |
| Vehicle body/mounts contaminate reconstruction | Static equirect mask from stage 1 |
| Visible cell-to-cell seams every ~50 m | Shared 10 m padding footage; if a seam survives, local fine-tune of the two boundary tiles; last resort: brief crossfade at cell boundaries only |
| 12 km batch: job failures mid-run | Resumable queue, per-cell status manifest, automated QC gate per cell |
| Storage blow-up | Compressed runtime format chosen in Phase 0; prune in `slice`; raw intermediates deleted per cell after QC pass |

## 8. Validation

- **Per stage, test section:** the visual examinations from the original draft are kept
  (trajectory smoothness, sparse-point alignment on trunks, chunk plots, render
  inspection, held-out views).
- **Per cell, automated (for 12 km):** held-out-view PSNR/SSIM threshold; splat count
  vs. cap; auto-rendered boundary strip images at every tile seam; short flythrough
  video per cell for spot checks.
- **On device:** frame timing (72 Hz sustained while streaming), memory watermark,
  load-latency vs. vehicle speed margin, boundary drive-through inspection.
- **ODGS side-experiment (Phase 3):** train one cell with ODGS on raw equirect frames,
  same poses/boundaries; compare foliage detail, stability, splat count, runtime perf.
  Not on the critical path.

## 9. Open questions

- Actual vehicle speed in footage (frames-per-10 m density) — measure in stage 1.
- Whether footage contains moving objects needing masking — review in stage 1.
- DGX Spark vs RTX 5060 Ti per-cell training throughput — benchmark in Phase 1.
