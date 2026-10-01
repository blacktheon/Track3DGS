# Local Stage 1 renderer extension

Upstream: <https://github.com/wuyize25/gsplat-unity>, revision
`a2bf458d6b16395e6570e9345f9f4408f92684b8`, package version `1.4.0`.
This package was embedded from Unity's pinned package cache on 2026-09-21.
The original `.meta` files and their GUIDs are preserved. `UPSTREAM_FILES.sha256`
records the SHA-256 of every original file before local modifications.

License: MIT. The original `LICENSE.md`, `Third Party Notices.md`, and source
copyright notices remain in place. Local additions use the same MIT license.

## Scope and contract

The Stage 1 selection extension retains the full uploaded source buffers.
An accepted rank contains unique storage IDs in importance order; its length M
may be less than the source count N because prior hard removal is represented by
omitting those IDs. Keep count k is an integer in [0,M]. IDs are compacted in
ascending storage order. For the current no-pruning PLY import, storage IDs are
original row IDs. A future reordered importer must provide an explicit identity
mapping before claiming canonical original-row tie ordering.

The GPU path uploads the inverse rank once, computes block counts and
hierarchical exclusive prefixes, then scatters the selected IDs. Every camera
copies that immutable list into the mutable depth-sort payload. Source buffers
are neither replaced nor reimported by a keep-count change. Cutouts and partial
uploads are incompatible with an exact Stage 1 rank; global merging must fall
back to per-renderer sorting while Stage 1 selection or wall masks are active.

Public renderer API: `SetStage1Rank(uint[])`, `SetStage1KeepCount(int)`,
`ClearStage1Selection()`, `Stage1SelectedCount`, `Stage1EligibleCount`, and
`HasStage1Selection`. Setting a rank initially requests all M eligible IDs.
Rank activation, clearing, and keep-count changes commit at the next renderer
update before draw submission, keeping that frame's selection, draw and sort
counts consistent. A replacement rank briefly owns separate selection buffers
until this boundary; slider changes do not allocate buffers. To display the
full eligible baseline during review, use k=M; clearing selection restores all N
source IDs, including IDs intentionally absent from the rank.

`Stage1GpuSelection` uses 256-thread count/scan/scatter groups and a recursive
block-prefix hierarchy. Its shader is under `Runtime/Shaders/Resources` so Unity
includes it in player builds. The two depth formats accept an explicit selected
count; existing shader tail guards reject slots after k. The sorter initializes
ordinary identity payloads before the first depth read and always sorts active
Stage 1 selections per camera, regardless of ordinary sort refresh settings.

Finite directional walls are a separate view-dependent visibility rule applied
to splat centers. Each wall stores a world-to-local matrix, bounds ordered
(xmin,xmax,zmin,zmax), a normalized world plane, and a rear depth in world units.
Only an eye with signed plane distance >1e-6 can block a center whose distance is
negative and at least -rearDepth. The eye-to-center segment must intersect the
finite local XZ rectangle, with edges included. Deeper centers remain eligible.
This is the agreed per-Gaussian visibility profile, not solid geometry.

Use `SetStage1Walls(Stage1Wall[])` and `ClearStage1Walls()` for up to 16 walls.
Wall transforms, normalized plane equations and depths are copied into the
renderer material property block. Spark and Uncompressed use the same vertex
visibility function. `Stage1Wall.BlocksView` is the CPU diagnostic equivalent.

## Validation

Tests live in `Assets/SplatPreprocess/Tests/EditMode/Stage1GpuSelectionTests.cs`.
The central existing Unity Editor runs the tests; no second Editor is launched.
The central initial RED run recorded all 21 new cases failing for missing
selection/wall behavior before production implementation. The first GREEN run
passed those 21 tests. A subsequent transition test reproduced immediate rank
replacement changing the depth input of an already submitted draw, before its
fix was added. The central `round1-lease-red.xml` run then passed all 22 renderer
tests, including that transition regression. Four expected missing-behavior
failures in that broader run were worker-lease tests outside this package.
Reports are under `SplatData/validation/m1/`.

GPU count and tie tests require the actual device to support the upstream
radix-sort wave operations. The renderer depth integration fixture uses
Uncompressed data; Spark uses the same selected-list and explicit depth-count
contract but its image/packing comparison is separate validation. Actual-source
image endpoints, Android and stereo evidence remain separate validation work.

## QuestSBTC editor pair preview, 2026-09-29

This embedded copy additionally changes `GsplatSorter.cs` and
`GsplatGlobalRenderer.cs` so compatible Spark pairs use one global sort and one
command-buffer draw in the URP editor preview. The inherited edit-mode path
previously disabled global merging and sorted the two renderers independently.
Camera layer masks apply when gathering editor renderers. Stage 1 selection,
directional walls and incompatible formats retain their existing fallback.

The regression helper in
`Assets/TrackTrainingPreview/Editor/TrackTrainingPreviewChecks.cs` creates two
assets with interleaved depths and compares their rendered colours with a
combined asset, facing forward and backward. Before the fix, global merging
was disabled and maximum RGB error was 0.200000. After the fix, merging was
enabled and maximum error was 0.000000 on this editor's D3D12 device.
This evidence does not establish Android or stereo performance. The
`UPSTREAM_FILES.sha256` file still identifies the original upstream contents,
not the locally modified package.
