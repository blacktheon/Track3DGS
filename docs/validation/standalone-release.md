# Standalone release verification — 1 October 2026

This release extracts the Step 1/2 work into Track3DGS and adds an independent
Unity viewer. It does not retrain or replace the six original Track02 regions.

## Evidence

- **CPU:** 129 tests passed in a freshly created Python 3.11.16 environment
  installed from `pipeline/requirements.txt`, with FFmpeg on PATH. Tests run
  from `pipeline` using `python -m pytest -q`.
- **Model 1 CUDA reproduction:** 378,428 source core rows; 1,932 centre seeds;
  iterative additions 2,642, 288, 14, 0; **373,552 retained**. The PLY and
  source-row provenance are byte-identical to the reviewed multi-view experiment.
  Forty before/after perspectives were rendered with the extracted evaluator.
- **Model 2 CUDA reproduction:** 612,278 raw rows; 4,095 initial candidates;
  iterative additions 47, 1, 0; **608,135 retained**. Provenance-based core
  transfer retains **434,740 of 437,360** rows. Both raw/core PLYs and provenance
  are byte-identical to the earlier single-camera result.
- **Independent Unity project:** a new Unity 6000.3.19f1 process imported the
  included URP/pinned renderer and ran synthetic route/model fixtures without
  QuestSBTC, XR, private packages or learned-model data. It compiled and verified
  route marker generation, asset counts/GUIDs/transforms, before/after camera
  preservation, two-slot limits, rejection of unavailable outputs, and GPU
  release when unloading.
- **Pair sorting:** global merging enabled; pair versus combined-reference
  maximum RGB error **0.000000** in forward and reverse synthetic views. Blank
  renders explicitly fail this check. Evidence uses D3D12, not Android/stereo.
- **Launcher:** the portable Visual Studio/CUDA discovery launched the actual
  RTX 5060 Ti GPU environment and the reproduced Model 2 cleanup.

The [environment snapshot](gpu-environment-2026-10-01.json) records installed GPU
package versions. The CUDA tests reused the working GPU environment; a complete
new-machine installation of the entire training stack was not performed. The
utility environment was installed afresh.

## Scope and limits

The standalone Unity integration check uses tiny synthetic PLYs so it can run
without publishing the capture dataset or loading several large models beside
an active editor. The real capture's original Unity previews and full-count
before/after inspection predate this extraction. Byte-identical native outputs
and translation-only cache tests connect that evidence to the portable code.

Visual quality, seamless joins and standalone Quest frame time are separate
acceptance tasks. The Track02 420 m join still requires ground-coverage review.
Sky cleanup remains experimental; the baseline reconstruction package exporter
does not silently substitute its results. Capture/model data and transient logs
remain local and Git-ignored.

Reproduce the CPU checks, [Unity fixture](../unity-viewer.md#self-test-without-a-capture),
and [sky recipes](../sky-cleanup.md#reproduce-the-two-track02-experiments) using the
documented commands. The real-data recipes require the original local inputs.
