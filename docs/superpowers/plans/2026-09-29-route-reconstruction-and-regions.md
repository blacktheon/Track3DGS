# Continuous Route Reconstruction and Regional Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task, according to the user's chosen execution method. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the tested Track3DGS pipeline to reconstruct a continuous video in one route frame, train and clean overlapping regions, validate their assembled boundaries, and export a portable reconstruction package for VR3DGS.

**Architecture:** Keep the existing section workflow and its successful algorithms. Add a versioned route workspace and resumable coordinator around reusable extraction, masking, COLMAP, Splatfacto and cleanup functions. Separate observation overlap from export ownership and publish immutable, validated packages.

**Tech Stack:** Existing Python utility environment; Python 3.11 GPU environment; FFmpeg; COLMAP 4.1.1; Nerfstudio 1.1.5 Splatfacto; gsplat 1.4.0; PyTorch 2.7.1+cu128; NumPy/SciPy/plyfile; PowerShell CUDA launcher.

**Spec:** [System design](../../architecture/pipeline-integration.md) and [asset package v1](../../contracts/asset-package-v1.md). The earlier [route proposal](../specs/2026-09-29-continuous-route-preprocessing-design.md) supplies rationale; the newer system boundaries and package contract govern interfaces.

**Status:** Written plan only. No tasks below are completed by creating this document. No reconstruction or training is launched by this planning task.

## Global constraints

- There are two reusable authoring projects. QuestSBTC is their first game consumer, not a third preprocessing tool.
- Training regions do not define the runtime layout.
- The first reconstruction input supplies video only.
- The initial route profile is 100 nominal-metre cores, 20 nominal-metre context, and 20 nominal-metre playable margins at each recording end.
- Keep best-of-three frame selection, eight yaws `[-135,-90,-45,0,45,90,135,180]`, 1600-square views, 100-degree FOV, COLMAP global mapper with overlap 48 and Splatfacto 30,000 iterations.
- Preserve the working Nerfstudio 1.1.5 / gsplat 1.4.0 / PyTorch 2.7.1+cu128 training environment. Do not upgrade dependencies or replace the trainer incidentally.
- Retain the existing section commands and defaults. New route defaults must not silently change old projects.
- Preserve `data/Export/Edited`, raw video, masks, calibration and completed source models. No destructive reset is part of resume.
- Portable packages use `gs-asset-package`, `schema_version: 1`, the contract's native-local PLY plus transform model, and explicit SH/colour conventions.
- No automatic claim of seamless output is allowed. Acceptance requires recorded visual review after ownership and boundary checks.
- This plan implements Track3DGS steps 1-2. VR3DGS features and QuestSBTC runtime belong to separate plans.

## Review focus

1. Variable-rate video, nonzero source timestamps and duplicate frame names must not corrupt cross-region identity. Covered by Task 2.
2. A long curve, hill, stop, reverse or disconnected SfM component must not become a straight/flattened or silently shortened route. Covered by Tasks 3-4.
3. A nontrivial export rotation must preserve SH3 appearance as well as positions and covariance. Covered by Tasks 1 and 5-6.
4. A hairpin, exact partition boundary or duplicate-looking foliage must not cause double ownership or indiscriminate point deletion. Covered by Tasks 4 and 7-8.
5. Cancellation, stale cache data, relocated packages or missing required files must leave accepted results intact and fail clearly. Covered by Tasks 9-10.

## Reuse versus modification

| Existing component | Reuse | Required change |
|---|---|---|
| `extract.py` | Sharpness scoring, best-of-group selection, FFmpeg integration | Add long-input batching and original PTS/identity metadata through a new route entry point |
| `automask.py`, `skymask.py`, `views.py` | Proven mask logic, manual mask reuse, 8-view geometry and union masks | Consume stable frame records; cache by source/settings hashes; do not rename masks independently |
| `track.py`, `colmap_export.py`, `trajectory.py` | Feature/matcher commands, global/incremental mapper, pinhole intrinsics, pose conversions and point hygiene | Route-wide ingestion, component coverage reporting, explicit scale modes and stable global frame |
| `level.py`, `mount_calibration.json` | Calibration evidence and transformation math | Apply one declared route transform; keep legacy per-section leveling isolated |
| `cells.py` | Core/context planning and COLMAP subsets | Configurable playable margins, global held-out split, full region list and branch ambiguity reporting |
| `train.py`, `cuda_env.bat` | Splatfacto settings, disabled normalization, export and alignment checks | Native export mode with explicit transform; resumable all-region execution; geometric and SH verification |
| `skyprune.py` | Sky projection, glitter and needle criteria | Native-local data plus transformed geometry; region-local observations; retained/removed row provenance |
| `slice.py` | Useful pure projection/pruning concepts | New ownership assembly consumes the cleaned source; no mandatory 10 m runtime tiles or blind endpoint cuts |
| `pack.py` | Existing legacy helpers and tests | New portable package writer; no automatic SH stripping or new independent leveling |
| `qc.py`, `evalpsnr.py` | Registration/count/alignment/image evaluation | Per-region and assembled-boundary reports; exact revision/camera/split references |
| `run_section.ps1`, `run_track01_batch.ps1` | Preserve as working legacy routes | New coordinator instead of invoking their cleanup and cell-0 assumptions |

The existing venv-only large-image-cache patch is a known dependency hazard. Record and test its behavior before long runs; make an idempotent compatibility check/patch available without changing the training recipe. The installed source, expected version and patch precondition must be checked explicitly.

## File map

Existing modules remain authoritative for their algorithms. New modules carry route policy and interchange, not duplicate implementations.

| New file | Responsibility |
|---|---|
| `pipeline/track3dgs/package_contract.py` | File references, transforms, PLY/provenance and package validation |
| `pipeline/track3dgs/route_config.py` | Route configuration and deterministic stage fingerprints |
| `pipeline/track3dgs/route_ingest.py` | Continuous-video extraction and stable frame identity |
| `pipeline/track3dgs/route_track.py` | Connected route reconstruction and one global calibration |
| `pipeline/track3dgs/route_regions.py` | Region planning and camera/branch coverage |
| `pipeline/track3dgs/route_training.py` | Per-region GPU process orchestration and clean model descriptors |
| `pipeline/track3dgs/route_assembly.py` | Unique ownership and retained-row assembly |
| `pipeline/track3dgs/route_qc.py` | Boundary views, comparisons and review/repair state |
| `pipeline/track3dgs/package_export.py` | Atomic reconstruction-package publication |
| `pipeline/track3dgs/route.py` | User-facing resumable coordinator CLI |
| `pipeline/check_training_environment.py` | Version/cache-patch validation for the working GPU environment |
| `pipeline/run_route.ps1` | Thin Windows launcher using the selected environments |

Common on-disk workspace: `data/routes/<route-id>/<revision>/` with existing `Project` subpaths for `frames`, `views`, `track`, `cells`, `train/export`, plus `route_config.resolved.json`, `route.json`, `cameras.jsonl`, `regions.json`, `state/`, `assembly/` and `reports/`. Keep source video external/read-only. A package export goes to an explicitly selected new directory.

Interfaces below use JSON-compatible dictionaries whose fields are specified here or in the contract. Frame records extend existing `name, src_index, t, sharpness` with `frame_id, source_sha256, source_pts_seconds, split`. `frame_id` combines the source hash and original decoded frame index. Internal names can remain `frame_000000` because one route workspace has one source video; cross-workspace identity never depends on that basename alone.

Each model descriptor contains `region_id, cell_index, raw_file, clean_file, source_sha256, source_count, retained_rows_file, local_to_package, sh_degree, radiance_encoding, alignment_report`. File paths in working descriptors are local; the package writer replaces them with portable FileRefs.

## Task 1: Contract validation and reference fixtures

**Files:** Create `pipeline/track3dgs/package_contract.py`, `pipeline/tests/test_package_contract.py`, and `pipeline/tests/fixtures/package_cases.json`. Use the mirrored contract; do not change existing internal manifest versions.

**Interfaces:** `file_ref(root: Path, path: Path) -> dict`; `validate_manifest(manifest: dict, expected_kind: str = "reconstruction") -> None`; `validate_package(root: Path, expected_kind: str = "reconstruction") -> dict`; `read_provenance(path: Path, count: int) -> np.ndarray`; `transform_geometry(xyz: np.ndarray, covariance: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]`.

- [ ] Write `test_relocated_package_resolves_relative_files`, `test_similarity_preserves_covariance`, `test_directional_sh_uses_asset_local_direction`, `test_provenance_is_packed_12_bytes`, and parametrized invalid-file/version/count/path/transform tests. Pin `len(provenance_bytes) == 12 * count`, `Sigma_out == s*s*R@Sigma@R.T`, and source-ID uniqueness across two input files with coincident centres. Include reflection/shear rejection and an SH3 rotation that would fail if only geometry moved.
- [ ] Run `..\.venv\Scripts\python.exe -m pytest tests/test_package_contract.py -q` from `pipeline`; confirm the new interface tests fail before implementation.
- [ ] Implement the strict reconstruction/common contract validator and deterministic fixture construction. Use small numeric fixture descriptions to generate temporary binary PLYs; do not commit a dataset. Document the same cases for the later VR3DGS adapter.
- [ ] Run the focused suite, then existing `test_pack.py`, `test_colmap_export.py` and `test_train.py`; expect passing invariants without changing legacy behavior.
- [ ] Review and commit this deliverable independently.

## Task 2: Continuous input and stable source identity

**Files:** Create `route_config.py`, `route_ingest.py`, `tests/test_route_config.py`, `tests/test_route_ingest.py`; modify `extract.py` only to expose reusable bounded extraction helpers. Preserve `run_extract` defaults and its callers.

**Interfaces:** `load_route_config(path: Path) -> dict`; `stage_fingerprint(stage: str, inputs: dict, settings: dict, tool_versions: dict) -> str`; `ingest_route(config: dict) -> Path` returns the route workspace. Config has `schema_version`, `route_id`, `revision`, `source_video`, `workspace`, `vehicle_mask`, `sky_prior`, `mount_calibration`, `capture`, `scale`, `reconstruction`, `regions`, `training`, and `toolchain`, as specified by the system design's example. Resolve input paths relative to the config file and store their hashes. Missing optional priors/calibration must be explicit nulls; a required vehicle mask cannot silently become all-white.

- [ ] Write tests asserting identical sharpness winners on the existing CFR fixture, preserved source PTS for VFR/nonzero-start fixtures, stable frame IDs across overlapping selections, no overwritten manual mask, and bounded extraction batches. Assert `frame_id` differs for equal basenames from different source hashes. Reject missing source and margins/config values with nonfinite or nonpositive lengths where inappropriate.
- [ ] Run `..\.venv\Scripts\python.exe -m pytest tests/test_route_config.py tests/test_route_ingest.py -q`; confirm failure before adding the new path.
- [ ] Implement route extraction while reusing `sharpness` and `select_sharp`. Keep temporary decode batches bounded; do not materialize the whole long video as temporary JPEGs. Carry real decoded presentation time, selection settings and mask identity forward.
- [ ] Run the new tests plus `tests/test_extract.py`, `tests/test_views.py`, `tests/test_skymask.py`, `tests/test_automask.py`; expect the old selection/mask/view behavior to remain intact.
- [ ] Review and commit. No real capture is required for these tests.

## Task 3: One connected route and explicit calibration

**Files:** Create `route_track.py`, `tests/test_route_track.py`; modify `track.py`, `level.py` and `colmap_export.py` through opt-in helpers rather than replacing section behavior.

**Interfaces:** `reconstruct_route(config: dict) -> Path` returns `route.json`; `build_route_metadata(frames: list[dict], camera_poses: list[dict], scale: dict, route_transform: np.ndarray) -> tuple[dict, list[dict]]`; `check_route_coverage(selected_frames: list[dict], registered_ids: set[str], components: list[dict]) -> dict`.

- [ ] Write synthetic curved, sloped and reversing trajectory tests. Assert one similarity transform preserves inter-frame geometry, relative height changes and directions; timestamps stay increasing while accumulated `s` stays nondecreasing. Verify a missing internal component is reported and prevents ready-for-training state. Test stationary/zero-length footage and inconsistent shared frame IDs. Verify nominal-speed scale is labelled approximate and applied once to the entire route, not separately by region.
- [ ] Run `..\.venv\Scripts\python.exe -m pytest tests/test_route_track.py -q`; confirm failures.
- [ ] Reuse current COLMAP command builders/global mapper and point filtering. Enumerate components before choosing a usable reconstruction. Keep source COLMAP and accepted route transforms as separate versioned evidence. Apply route-wide approximate calibration without independently reheading future cells. `manual`, `nominal_speed` and `unscaled` modes obey the system spec; do not reinterpret unknown units as metres.
- [ ] Produce registration/coverage JSON plus top-down and elevation previews. Record disconnected spans and uncertainty; do not generate a false interpolated road across a failed interval. Defer a windowed/hierarchical mapper backend unless the real route demonstrates the need, preserving the same output contract.
- [ ] Run focused tests plus `test_track.py`, `test_level.py`, `test_trajectory.py`, `test_colmap_export.py`. Review and commit.

## Task 4: Region cores, training context and photographic holdouts

**Files:** Create `route_regions.py`, `tests/test_route_regions.py`; extend `cells.py` using an opt-in route planner. Retain the old `plan_cells` behavior/tests.

**Interfaces:** `plan_regions(route: dict, cameras: list[dict], settings: dict) -> dict`; `write_region_subsets(workspace: Path, plan: dict) -> None`. A region has `region_id` (`cell_000`, etc.), `cell_index`, `core_s`, `context_s`, `train_camera_ids`, `held_out_camera_ids`, `boundary_ids`, and `coverage_state`.

- [ ] Write tests for a final partial region, exactly-on-boundary camera/point, too-short playable interval, sparse boundary coverage and a nearby nonadjacent route branch. Assert core intervals cover the playable interval once, context overlaps intentionally, and every tenth selected frame's eight crops are excluded from Gaussian training together. Assert COLMAP subset poses equal the global accepted poses without re-normalization.
- [ ] Run `..\.venv\Scripts\python.exe -m pytest tests/test_route_regions.py -q`; confirm failures.
- [ ] Add 100/20 nominal-metre planning and configurable capture margins. Store continuous-polyline segment IDs and branch-ambiguity diagnostics. Use context range as the initial observation set; add visibly useful cameras only with a recorded rule. Use the existing subset writer and spatial point filtering. Keep photographic verification membership global so a view cannot become training data in one neighboring region and a claimed independent test in another.
- [ ] Run focused tests plus `test_cells.py` and `test_views.py`. Review the plan's core/context and split report, then commit.

## Task 5: Native-frame export and resumable regional training

**Files:** Create `route_training.py`, `check_training_environment.py`, `tests/test_route_training.py`; modify `train.py` with a keyword-only `export_frame="legacy"` mode. Add `tests/test_training_environment.py`.

**Interfaces:** `run_region_training(workspace: Path, region: dict, settings: dict) -> dict` returns the model descriptor; `check_training_environment(python: Path, apply_cache_patch: bool = False) -> dict`. Existing `run_train` keeps its default behavior; route mode explicitly requests `export_frame="native"`.

- [ ] Write tests asserting all region indices get distinct commands/checkpoints, the native PLY retains every SH3 field unchanged, and its `local_to_package` reproduces the existing geometry-alignment convention plus declared global calibration. Verify directions transformed back to the PLY's frame match the reference SH evaluation. Test interruption after region 1: rerun skips only hash-matching completed region 1, then executes regions 2 and 3. A changed pose/config hash must invalidate reuse.
- [ ] Run `..\.venv\Scripts\python.exe -m pytest tests/test_route_training.py tests/test_training_environment.py -q`; confirm failures with mocked processes (no GPU training).
- [ ] Reuse Splatfacto commands, 30k iterations, masks and disabled pose normalization. Read and record `dataparser_transforms.json`; verify the actual export basis instead of relying on an unexplained new Euler correction. Preserve native PLY bytes in route mode and represent correction in metadata. Maintain the legacy corrected export separately.
- [ ] Validate the installed training versions and known cache-pinning patch; an optional repair must be idempotent and refuse an unexpected installed source. Keep one GPU subprocess active. Record progress/checkpoint identity and process failure; do not run `run_section.ps1` to perform route cleanup.
- [ ] Run the focused suites and `test_train.py`. Perform one small, explicitly bounded GPU environment/export smoke check during implementation, with measured geometry and SH reference renders before a real route batch. Review and commit.

## Task 6: Reuse cleanup with coordinate and row provenance

**Files:** Extend `route_training.py` and `skyprune.py`; create `tests/test_route_cleanup.py`. Reuse existing pure mask/needle functions and keep legacy `run_skyprune` defaults.

**Interfaces:** `clean_region(workspace: Path, region: dict, model: dict, settings: dict) -> dict` returns a model descriptor with `clean_file` and `retained_rows_file`; original raw file/count/hash remain unchanged. Retained row indices are uint64 internally for the new exchange path.

- [ ] Write tests for transformed centres and scale-aware needle thresholds, shared-camera mask selection, a road that rises/falls, and SH3 record preservation. Assert the cleaned records exactly equal the selected raw rows and that original row indices survive successive filters. Verify unavailable/unmatched masks cause an explicit diagnostic rather than a false successful no-op.
- [ ] Run `..\.venv\Scripts\python.exe -m pytest tests/test_route_cleanup.py -q`; confirm failures.
- [ ] Evaluate masks and physical thresholds in the declared common frame while retaining PLY-local attributes. Limit comparisons to appropriate region/context observations. Use route-local camera height context for the existing canopy/glitter rule so one long-route mean height does not remove valid elevated road content. Preserve original formulas where applicable and record settings. Do not apply the legacy short-clip end-cut at every internal region boundary.
- [ ] Run focused tests and `test_skyprune.py`. On the small real pilot, compare raw/removed/retained views against the existing cleanup behavior; record any changed classification and its coordinate/coverage cause. Review and commit.

## Task 7: Unique ownership and non-destructive assembly

**Files:** Create `route_assembly.py`, `tests/test_route_assembly.py`; reuse suitable pure geometry helpers from `slice.py`, leaving legacy slicing unchanged.

**Interfaces:** `assign_region_owners(points_package: np.ndarray, route: dict, plan: dict, overrides: list[dict]) -> tuple[np.ndarray, dict]`; `assemble_route(workspace: Path, plan: dict, models: list[dict], overrides: list[dict]) -> dict`. Assembly records list each selected model/row subset, owner, transform and source identity; they are not renderer tiles.

- [ ] Write tests asserting an exact boundary uses the next core except the final endpoint, every exported identity has one owner, and the input clean/raw files are unchanged. Preserve two legitimate nearby splats with distinct source identities. Compare a world-coordinate duplicate from two neighboring models: only the selected region exports its representation. A hairpin with ambiguous nearest branches must require a declared spatial override or shared-region solution, not silently pass.
- [ ] Run `..\.venv\Scripts\python.exe -m pytest tests/test_route_assembly.py -q`; confirm failures.
- [ ] Implement continuous-polyline projection, deterministic ownership, explicit spatial overrides and an assembly manifest. Use the cleaned model descriptor, never default back to `cell_000.ply`. Retain full Gaussian footprints. Report missing owners/empty unexpected cores, context-discard counts and ambiguity. Do not equate identity checks with visual seam correctness.
- [ ] Run focused tests and `test_slice.py`, `test_pack.py`. Review the assembly report and commit.

## Task 8: Boundary verification and bounded repair workflow

**Files:** Create `route_qc.py`, `tests/test_route_qc.py`; extend `qc.py` and `evalpsnr.py` with optional per-model transforms and region selection. Preserve section defaults.

**Interfaces:** `make_boundary_views(route: dict, plan: dict, camera_profiles: list[dict], seed: int) -> list[dict]`; `evaluate_assembly(workspace: Path, assembly: dict, views: list[dict]) -> dict`; `plan_boundary_retry(plan: dict, boundary_id: str, attempt: int) -> dict`; `validate_review_record(review: dict, report_hash: str) -> None`.

- [ ] Write tests for deterministic camera generation, no held-out/training frame leakage, transformed multi-asset compositing and stale acceptance rejection. Assert unresolved boundaries remain `review_required`; a report without numeric errors is not automatically visually accepted. Verify context grows by 10 nominal metres per retry, attempts beyond two stop automatically, and capture limits remain respected.
- [ ] Run `..\.venv\Scripts\python.exe -m pytest tests/test_route_qc.py -q`; confirm failures.
- [ ] Generate JSON/HTML comparisons covering each seam from both directions, lateral offsets and varied headings. Photographic held-outs compare to real images; synthetic comparisons against individual parent regions are labelled diagnostic, not ground truth. Inspect colour, alpha, silhouettes, holes and duplicate structures with surrounding content rendered together.
- [ ] Implement bounded context-expansion retry through the same region trainer. Persist any failed seam for review. A shared-boundary training region can be an explicit repair request using the same production backend and fixed frame; validate all replacement joins. Do not invent automatic success thresholds or silently introduce the rejected custom MCMC trainer. Advanced joint refinement is a separately evaluated follow-up if the production backend cannot resolve a seam.
- [ ] Run CPU tests, then real boundary renders on the three-region pilot. Record exact images, settings, hashes and human visual review before acceptance. Review and commit the implementation independently of dataset acceptance.

## Task 9: Portable reconstruction export

**Files:** Create `package_export.py`, `tests/test_package_export.py`; update documentation examples. Do not alter legacy `pack.py` output to masquerade as the new schema.

**Interfaces:** `export_reconstruction(workspace: Path, assembly: dict, review: dict, output: Path, state: str = "draft") -> Path` returns the new package manifest; it uses Task 1's validator and FileRef/provenance routines.

- [ ] Write relocation, missing-file, tampered-hash, source-collision and incomplete-staging tests. Assert preserved SH degree/attributes, `12*count` provenance bytes, relative package paths, native transform correctness, and exact retained counts. An accepted export must reject unresolved/stale seam review; a draft export lists limitations explicitly. Include an external-source fixture with no route to ensure the shared contract remains general.
- [ ] Run `..\.venv\Scripts\python.exe -m pytest tests/test_package_export.py -q`; confirm failures.
- [ ] Copy selected original rows into region PLYs, generate source registry/provenance, package shared transforms/camera/route metadata and reports, then validate and atomically publish into a new directory. No automatic SH stripping, independent re-leveling, absolute machine paths or Unity GUID references. Original image evidence is an explicit optional attachment, not mandatory bulk data.
- [ ] Run focused tests and contract fixtures from a relocated temporary directory. Verify generic contract fields without depending on installed Unity or the other repository. Review and commit.

## Task 10: Coordinator, documentation and the real pilot

**Files:** Create `route.py`, `pipeline/run_route.ps1`, `tests/test_route_runner.py`, `docs/route-workflow.md`, and `pipeline/examples/route_config.example.json`; update `README.md`, `docs/handover/05-current-state-and-next-steps.md` and the command reference with accurate implemented status.

**Interfaces:** `run_route(config_path: Path, through: str, resume: bool = True, dry_run: bool = False) -> dict`; stages are `ingest`, `track`, `regions`, `train`, `assemble`, `verify`, `export`. `train` includes per-region cleanup. CLI: `python -m track3dgs.route --config <file> --through <stage> [--dry-run] [--no-resume]`. `--no-resume` requires a new revision directory; it is not permission to delete an existing one.

- [ ] Write mocked end-to-end tests asserting every region is processed, dependencies invalidate in order, unchanged regions are reused, failed stages prevent acceptance/export, and dry-run starts no subprocess. Test cancellation between subprocesses and stale/interrupted state on restart. Assert `data/Export/Edited` and prior outputs are untouched.
- [ ] Run `..\.venv\Scripts\python.exe -m pytest tests/test_route_runner.py -q`; confirm failures.
- [ ] Implement atomic stage state with input/config/tool hashes and resume checks. Resolve the configured utility/GPU interpreters; run the existing CUDA setup only for GPU stages. Surface logs/progress and truthful partial status. Add an example config containing explicit scale mode, source path, 100/20 region settings, margins and separate toolchain paths; label its assumed speed as an example, not a measured capture value.
- [ ] Run the full utility suite from `pipeline`: `..\.venv\Scripts\python.exe -m pytest -q`. GPU-dependent checks run in the declared training environment and are reported separately. Compare the documentation's proposed CLI to the actual help output.
- [ ] On the supplied continuous video, reconstruct the whole candidate route, inspect approximate scale/shape/elevation/coverage, then process only three adjoining regions first. Choose a bend/slope and complex foliage if present. Record peak GPU/RAM/disk use, elapsed times, actual region counts and all boundary outcomes; do not extrapolate them into a Quest performance claim.
- [ ] Export a validated pilot package and inspect it through a minimal reference viewer/contract fixture path. A future VR3DGS package adapter is a separate milestone; do not claim full application interoperability before it exists. Record any remaining integration requirement explicitly.
- [ ] Review and commit the coordinator/docs. Mark the route pilot accepted only after actual visual approval; no successful unit-test count substitutes for that decision.

## Acceptance checklist for steps 1-2

- [ ] A single continuous-video test has one connected route frame with explicit approximate calibration and inspectable bends/elevation.
- [ ] All intended source intervals have a documented coverage status; no disconnected component is silently discarded.
- [ ] Three neighboring regions are trained, cleaned and assembled from shared poses with reproducible observation splits.
- [ ] Ownership and row provenance pass; Gaussian footprints are not clipped and raw/edited originals remain recoverable.
- [ ] Every playable join has reviewed evidence or the package is clearly draft with the failed join listed.
- [ ] Native-local SH3 appearance and geometry pass the coordinate/colour fixtures; geometric alignment alone is insufficient.
- [ ] Resume/cancellation do not corrupt accepted predecessors or retrain unchanged complete regions.
- [ ] A relocated reconstruction package validates without either Unity project or the producer's absolute working paths.
- [ ] The working section workflow and its tests still pass. Historical numerical quality is compared on a bounded reference case when a reused algorithm's behavior changes.

## Deferred work

Registering all legacy edited PLYs, a new hierarchical SfM backend, a replacement trainer, automatic joint seam optimization beyond the production-backend pilot, generic VR3DGS UI/SH3 integration, LOD1-3 generation, plugin replacement and QuestSBTC streaming are not hidden tasks in this plan. Each is activated by the relevant measured need or its own implementation plan. Preserve the legacy source/pose/edit data so migration remains possible.

Execution should begin with Task 1 after the design/plan review. The tasks are mostly sequential because their contracts feed one another; inline execution is a reasonable default if the user asks to proceed without selecting a delegation method.
