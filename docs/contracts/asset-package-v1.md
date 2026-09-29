# Gaussian asset package v1

Status: Proposed interchange contract, 2026-09-29. No current importer/exporter is claimed to implement this complete contract. This is separate from the existing `source_manifest.json`, `rank_manifest.json`, `stage2_input.json` and Track3DGS tile manifest formats.

Canonical specification: this file in the VR3DGS repository. Track3DGS carries an identical copy so its implementation plan is self-contained. Change the contract and both copies together; future producer/consumer releases must record the contract revision and run the same fixtures. No third authoring repository is required.

## 1. Purpose and package types

A package is a directory containing `manifest.json` and its referenced data. A ZIP may transport the directory, but consumers validate and extract it before import. Python environments, Unity scenes/GUIDs and runtime renderer buffers are not exchange formats.

| `kind` | Producer | Consumer | Meaning |
|---|---|---|---|
| `reconstruction` | Track3DGS or another producer | VR3DGS Stage1 | Coherent source assets; route metadata is optional |
| `accepted` | VR3DGS Stage1 | VR3DGS Stage2, optionally an application | User-accepted retained representation and source provenance |
| `lod` | VR3DGS Stage2 | QuestSBTC or another application | Immutable LOD0 reference plus per-chunk representations |

VR3DGS must also accept a compatible standalone PLY through an import adapter. The adapter creates equivalent metadata after the user declares coordinates, scale and appearance conventions. A model does not need a route, a video, cameras, a NavMesh, a tank or Track3DGS naming conventions to be a valid source.

## 2. Common manifest

Required top-level fields:

| Field | Type / rule |
|---|---|
| `format` | Literal `gs-asset-package` |
| `schema_version` | Integer `1` |
| `kind` | `reconstruction`, `accepted` or `lod` |
| `package_id` | UUID string for the logical asset family |
| `revision` | Nonempty immutable revision string, distinct within the family |
| `created_utc` | ISO-8601 UTC timestamp |
| `producer` | `{name, version, git_commit}`; commit may be null for external tools |
| `parents` | Array of `{package_id, revision, manifest_sha256}`; empty for a new source |
| `required_capabilities` | Array of feature IDs the consumer must support |
| `coordinate_system` | The declaration in section 3 |
| `sources` | Original/derived source registry from section 5 |
| `assets` | Gaussian asset records from section 4 |
| `validation` | `{state, reports, limitations}`; state is `draft` or `accepted` |

Type-specific `reconstruction`, `stage1` or `lod` is required for the corresponding kind. `presentation` and `extensions` are optional. Omit absent optional sections rather than inventing camera data. JSON uses UTF-8, finite numbers, and no duplicate keys. IDs are strings and counts are nonnegative integers. Current v1 consumers may reject individual PLYs above 2,147,483,647 rows and must report that limit explicitly.

A **FileRef** is `{path, sha256, bytes}`. `sha256` is 64 lowercase hex characters and covers the file's raw bytes. `path` uses forward slashes and is relative to the package root. Absolute paths, drive letters, `..`, and resolved paths or archive links escaping the root are invalid. Paths must also be unique under case-insensitive comparison for Windows compatibility. Verify hashes and lengths before import.

Packages are immutable. Build into a staging directory, verify it, then publish the completed directory. No consumer imports partial/staging data. A manifest's raw-file SHA-256 is the package revision identity used by consumers; it is recorded externally or by child manifests, not as a self-referential hash inside itself. Producers must not reuse a family/revision pair with different bytes.

Unknown schema versions or unsupported required capabilities are errors. Optional unknown `extensions` may be retained and ignored. Required capabilities include `ply.sh0` through `ply.sh3` as applicable, `transform.similarity`, `provenance.u32-u64`, `lod.independent-ply` for LOD sets, and `presentation.glb` when visual meshes are necessary to the accepted appearance. Extra capability IDs require an explicit consumer implementation.

## 3. Coordinates, scale and appearance

The package frame is **right-handed, right/up/back (RUB)**. Its origin and horizontal orientation are chosen once and documented. A coordinate declaration contains:

```json
{
  "axes": "RUB",
  "handedness": "right",
  "matrix_layout": "row-major",
  "vectors": "column",
  "unit": "meter",
  "scale_status": "approximate",
  "scale_basis": "One global factor from an explicitly assumed average speed",
  "level_status": "approximate"
}
```

`unit` is `meter` or `unit`; `scale_status` is `calibrated`, `approximate` or `unscaled`. `unscaled` requires `unit: unit`; calibrated/approximate metre frames use `unit: meter`. `level_status` is `calibrated`, `approximate` or `unknown`. A numerical distance in an approximate metre frame is nominal, not a surveyed measurement. Each change to calibration creates a new revision and invalidates dependent ranks/layouts/views.

Each Gaussian asset has its own right-handed local frame and a 16-number `local_to_package` matrix. The matrix is applied to column vectors: `p_package = T * [p_local, 1]`. V1 permits finite, invertible **positive uniform-scale similarity transforms**, with homogeneous last row `[0,0,0,1]`. Shear, reflection and nonuniform scale must be rejected or converted through a separately validated import adapter. This avoids silently misrepresenting anisotropic splats.

Keeping the PLY in its native frame preserves its Gaussian and SH records. For `T = [sR,t]`, positions transform as `sR*p+t`, covariance as `s² R*Sigma*Rᵀ`, and directional colour is evaluated in the original asset frame using `d_local = Rᵀ*d_package`. Translation does not change SH coefficients. A producer that bakes rotation into PLY geometry must also rotate the SH basis correctly, or use SH0 explicitly; merely leaving higher-order coefficients untouched changes appearance.

For Unity, the package-to-Unity basis change is `F = diag(1,1,-1,1)`. If a renderer importer already applies `F` to asset-local data, the Unity object transform is `F*T*F`; do not apply the reflection twice. This is an adapter responsibility, verified with an asymmetric geometry fixture and directional SH fixture. A renderer that bakes everything into world buffers may use a different internal representation if these tests agree.

Camera records use an OpenCV-style local camera frame: +X right, +Y down, +Z forward. `camera_to_package` maps that frame into the package frame. Projection uses pixel coordinates with top-left image origin, pixel centres at `(0.5,0.5)`, and explicit `fx,fy,cx,cy,width,height`. Runtime clip planes are application settings, not inferred from the source video.

## 4. Gaussian PLY records

V1's mandatory interchange codec is `ply-3dgs-f32-le`: binary little-endian PLY 1.0, one vertex element, scalar float32 fields, no quantization. A vertex requires:

| Fields | Interpretation |
|---|---|
| `x,y,z` | Centre in asset-local coordinates |
| `rot_0..rot_3` | Nonzero quaternion in **w,x,y,z** order; normalize for evaluation |
| `scale_0..scale_2` | Natural logarithms of positive Gaussian standard deviations |
| `opacity` | Logit; evaluate with sigmoid |
| `f_dc_0..f_dc_2` | Real SH degree-zero coefficients for R/G/B |
| `f_rest_0..` | Optional complete SH bands, degree 1-3 |

SH uses the conventional 3DGS real basis with camera-to-splat direction in asset-local space. Higher-order fields are channel-major: all non-DC coefficients for R, then G, then B. Degree 0/1/2/3 has 0/9/24/45 `f_rest` fields. Degree-zero colour before renderer colour conversion is `0.5 + 0.28209479177387814*f_dc`. `radiance_encoding` explicitly declares `srgb` or `linear`; adapters match the reference pipeline and must not apply an extra colour-space conversion. It describes the RGB domain the SH model was fitted in, not the coordinate basis.

Validate header length, exact row count/file length, finite attributes, usable quaternion norms and finite positive decoded scales. Reject invalid source data without silently dropping rows. Additional scalar float32 properties may be preserved, but required semantics must not be guessed from arbitrary point-cloud properties. If an algorithm cannot preserve an auxiliary field after merging, it reports its removal in provenance rather than inventing values.

Each asset record requires:

```text
asset_id: unique string in this revision
file: FileRef
codec: "ply-3dgs-f32-le"
count: integer
sh_degree: 0 | 1 | 2 | 3
radiance_encoding: "srgb" | "linear"
local_to_package: 16 finite numbers
provenance: FileRef
render_bounds: {min: [x,y,z], max: [x,y,z], support_sigma: 4.0}
```

Bounds are in package coordinates and include covariance extent using the declared support cutoff. They are not just centre bounds. Renderers must additionally allow for their screen-space filter/antialias footprint; four-sigma metadata alone is not an exact visibility guarantee for every projection/backend. Portable data does not contain renderer-specific compressed buffers. Those belong to application build caches.

## 5. Identity and provenance

`sources` is an ordered registry of `{source_id, kind, sha256, count, producer}`. `source_id` is stable within the asset family; `kind` is `original` or `derived`. `sha256` addresses that source's original PLY bytes; a derived entry also records its parent package revision and generation settings in `producer`. Original files need not be redistributed, but their hashes/counts remain available.

Every PLY row has one packed 12-byte provenance record: little-endian `uint32 source_index`, followed immediately by little-endian `uint64 source_row`, with **no padding**. `source_index` refers to this package's registry; `source_row` is zero-based and below the registered count. Provenance length must equal `12 * asset.count`. The stable identity is `(source_id, source_row)`, not the package-local registry index. This avoids collisions between independent input files and JavaScript/JSON uint64 rounding.

Stage1 row selection preserves original records and identities. Several assets may cite the same source, but the accepted representation must not contain duplicate identities. LOD0 partitions those accepted identities exactly once. LOD1-3 may create new Gaussians; register derived sources rather than claiming their rows are unchanged originals. Optional many-to-many parent mappings can support auditing/refinement, but are not required to render v1.

Preserve the current internal uint32 source maps where appropriate; convert at the exchange boundary. Do not reinterpret their bytes as the new 12-byte layout.

## 6. Reconstruction package: Track3DGS to VR3DGS

Example layout:

```text
route-r001/
  manifest.json
  regions/region-000.ply
  regions/region-000.provenance.bin
  regions/region-001.ply
  regions/region-001.provenance.bin
  metadata/regions.json
  metadata/ownership.json
  metadata/route.json
  metadata/cameras.jsonl
  reports/reconstruction.json
  reports/seams.json
```

The required `reconstruction` object contains FileRefs `regions`, `ownership` and `quality_report`. Track3DGS also supplies `route`, `cameras` and `seam_report`. Another producer may omit those track-specific files and state its own limitations. VR3DGS must not reject an ordinary object because it has no route.

`regions.json` contains `region_id`, referenced `asset_ids`, `core_s`, `context_s`, observation IDs and `boundary_ids`. Route intervals are half-open except the last interval includes the route endpoint. Camera intervals may overlap; export ownership may not. These regions are training/authoring units, not mandatory runtime chunks.

`ownership.json` declares the partition version, region owners, boundary convention and any explicit spatial overrides. Ordinary open-route assignment can use closest-polyline arc length. Ambiguous nearby nonadjacent route branches must be flagged and resolved by a spatial override or shared region; silently assigning all of a hairpin by nearest sampled point is not allowed. Unique centre ownership does not clip a Gaussian's footprint or prove visual seam quality.

`route.json` contains `schema_version: 1`, `route_id`, source-video hash, scale basis, playable interval, capture margins and a `samples` array. A sample records `frame_id`, `timestamp_seconds`, `s`, and `rig_to_package`. Timestamps reference the original decoded source presentation timeline; `s` is monotonic accumulated distance in package units. The trajectory is optional reference geometry, not an application driving constraint.

Each `cameras.jsonl` record contains `camera_id`, `frame_id`, `timestamp_seconds`, `camera_to_package`, intrinsics, `split` (`development` or `held_out`) and optional image/mask FileRefs. Images and masks may be omitted to keep the asset handoff small. They are required in a separate optional training-evidence attachment when a later refinement task needs original photography; missing images disable that capability explicitly. Synthetic reference views are still possible without them.

Track3DGS may export `validation.state: draft` for review with unresolved seams clearly listed. `accepted` requires completed ownership checks, resolved boundaries and recorded human visual acceptance. Automated numeric metrics alone do not grant visual acceptance. VR3DGS may inspect a draft but cannot silently promote it to a validated route baseline.

## 7. Accepted package: Stage1 to Stage2

The required `stage1` object records the parent package or external import revision, view-domain hash, camera-profile hash, ranking hash, manual-exclusion revision, reference counts, retained counts and acceptance metadata. Each source/region has exact original, excluded, eligible and retained counts. The retained fraction is explicitly `retained / eligible`; zero eligible rows require a null fraction, not division by zero.

Include FileRefs for available importance/rank data and the accepted view-domain description under `stage1.evidence`. These are authoring evidence, not runtime dependencies. The generic view-domain schema can hold camera poses directly or NavMesh triangles plus camera-rig constraints; tank-specific interpretation belongs to an optional profile.

Assets contain only accepted rows and retain their transforms/appearance declarations. Optional replacement visual surfaces travel through section 9. The package does not reference a Unity material GUID or an absolute path in `SplatData`.

## 8. LOD package: Stage2 to applications

The required `lod` object includes `layout_id`, `accepted_parent`, a FileRef `chunks`, FileRef `quality_report`, and an optional FileRef `review_cameras`. `accepted_parent` identifies the exact accepted manifest hash. All Gaussian assets for every level are listed in common `assets`.

`chunks.json` uses `schema_version: 1` and a `chunks` array. Each chunk has:

| Field | Meaning |
|---|---|
| `chunk_id` | Stable within the layout revision |
| `partition_bounds` | Nonoverlapping spatial ownership cell, in package coordinates |
| `render_bounds` | Conservative union of all available level bounds |
| `lods` | Sorted array of `{level, asset_ids, count, quality_profile_id}` |
| `group` | Optional authoring group; no mandatory `machine` or `track` value |

Levels are integers 0-3; LOD0 is mandatory. A level may reuse the same asset IDs when further reduction is meaningless. Counts must be non-increasing and equal the sum of referenced asset counts. A complete four-level deliverable lists all four levels; an LOD0-only package declares only 0 and cannot advertise ready lower levels.

Each LOD0 asset belongs to exactly one chunk, and its source identities cover the accepted input once. Coarse levels may have different geometry but retain the same package frame. Do not merge unrelated surfaces solely to meet a count target. Validate neighboring mixed levels with surrounding context and correct per-camera transparent sorting.

Quality profiles describe the reference camera set, FOV/resolution, sampled RGB/alpha/silhouette errors, intended viewing conditions and unresolved findings. Sampled image error is not a universal mathematical error bound. Universal distance thresholds or a fixed three-region budget are not part of the package. The application chooses levels based on its view and measured cost.

## 9. Optional visual surfaces and auxiliary data

`presentation` contains `objects`: `{object_id, file, local_to_package, purpose}`. `file` references a self-contained glTF 2.0 binary `.glb`; `purpose` is `visual`. Embedded textures/materials use a declared supported profile: opaque PBR metallic-roughness or `KHR_materials_unlit`. Add the appropriate required capability if a material needs more. Arbitrary Unity shaders and GUID references are not portable.

glTF assets use their defined right-handed local coordinates; their explicit matrix establishes placement in the package frame. Preserve the accepted baked planes and textures as actual geometry/material data. A consumer must not silently omit required presentation content because it only supports splats. Collision geometry, navigation and game scripts are separate optional application inputs; splats and visual planes do not automatically supply physics.

Authoring annotations, quality marks and suggested view domains can be carried in extensions with versioned schemas. Unknown required behavior is rejected. The earlier paused headset-recording feature is not a prerequisite for these packages.

## 10. Application import and feedback

QuestSBTC validates the package, then builds renderer-specific assets using an adapter keyed by `(manifest_sha256, adapter_id, adapter_version, target_platform, compression_profile)`. This cache is disposable and must not replace the portable master. Cached counts, transforms, SH handling, visual meshes and bounds are checked against the master.

The runtime may stream assets and share resident buffers across cameras, but depth ordering is evaluated correctly for each camera. Active representation transitions must not leave holes or double-opacity geometry. The game owns thresholds, prefetch, memory limits and multiplayer-local camera selection.

Optional `runtime-report.json` feedback contains `schema_version: 1`, package/layout IDs and manifest hash, application/build/adapter identifiers, device/graphics API, camera profiles, selected/resident/rendered counts, CPU/GPU frame-time distributions, memory peaks, streaming stalls and thermal-test duration. Missing measurements are null with reasons, not zero. A quality mark references a chunk/layout/revision and a reproducible camera; it need not involve continuous headset recording.

## 11. Compatibility and acceptance fixtures

Before declaring interoperability, both authoring projects and the chosen game adapter must pass common fixtures:

1. An asymmetric SH0 scene tests axes, scale, covariance and one-time Unity handedness conversion.
2. A directional SH3 scene with a nontrivial asset transform tests coefficient order, direction and colour conversion.
3. A two-source scene with coincident centres tests unique source identity without deleting legitimate nearby Gaussians.
4. A pruned/partitioned scene tests byte-preserved retained records, provenance and all-LOD0 equivalence.
5. A visual plane with an embedded texture tests presentation retention.
6. Unsupported version/capability, modified file, duplicate ID, wrong count, singular transform and path escape fixtures test clear rejection.

These are planned conformance fixtures. Existing project-specific tests are useful starting points but do not establish that the new contract already works.
