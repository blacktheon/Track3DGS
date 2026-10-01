# Standalone Track3DGS Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish self-contained Steps 1 and 2, including the Unity review tools.

**Architecture:** Keep the existing Python route pipeline. Extract ignored cleanup experiments into reusable modules and provide a minimal Unity viewer with the tested embedded renderer. Generated data stays outside Git.

**Tech Stack:** Python 3.11+, NumPy/OpenCV/gsplat, Unity 6000.3.19f1, URP 17.3.0.

**Spec:** `docs/superpowers/specs/2026-10-01-standalone-release.md`

## Global Constraints

- No QuestSBTC gameplay dependency or assets; no capture/model data in Git.
- Preserve original PLYs, full SH, provenance and existing user edits.
- At most two visible Unity models; cleanup results remain experimental.
- Reuse the existing isolated Track3DGS worktree; fast-forward master after verification.
- Explicit user authorization covers implementation, commit and normal push.

## Review Focus

- A new checkout must not require workstation-specific executable paths (Task 3).
- Overlapping yaw views must not count as independent positions (Task 1).
- Sky and vehicle masks must remain distinct; held-out frames must not leak into edits (Task 1).
- A stale route/catalog or changed source must not silently reuse incorrect model placement (Task 2).
- Unity must compile and render without QuestSBTC, including paired global sorting (Task 2).

### Task 1: Portable sky cleanup and diagnostics

**Files:** `pipeline/track3dgs/sky_volume*.py`, `route_lateral_qc.py`, `pipeline/tests/test_sky_volume.py`, `pipeline/configs/sky-*.json`.

**Interfaces:** Consumes route cameras, masks and native PLY/provenance. Produces `selection.json`, `models/sky.ply`, `processing.json`, comparison images and HTML; all under an explicit new output directory.

- [ ] Write/run tests for top-connected sky, calibrated projection, ellipse spikes, distinct-position consensus, selection separation and exact retained rows. Expected: missing module/API fails.
- [ ] Extract parameterized prepare/carve/evaluate commands and the lateral diagnostic; preserve both original experiment strategies.
- [ ] Run CPU tests and reproduce Model 1 with the existing data in a new output directory. Expected: all tests pass; retained rows/count match the prior result.
- [ ] Commit algorithm extraction with reproduction settings and limitations.

### Task 2: Independent Unity viewer

**Files:** `unity/Track3DGSViewer/`, `route_preview_models.py`, preview publisher tests.

**Interfaces:** Consumes native masters and Task 1 cleanup outputs. Publishes one route-bound `Assets/Track3DGSData/catalog.json` plus translation-only caches and `route_preview.json`.

- [ ] Test generic route publication, source/route mismatch rejection and sky camera conversion. Expected: new publication cases fail.
- [ ] Copy/refactor the three Unity editor scripts and pinned renderer; add minimal URP setup and standalone smoke fixtures.
- [ ] Verify in a separate Unity process with generated tiny data and actual two-model data, preserving the user's open project. Expected: compile, route markers, two-slot load/clear, camera preservation and pair-sort checks pass.
- [ ] Commit viewer, publisher and dependency notices.

### Task 3: Setup and public documentation

**Files:** `README.md`, `docs/setup.md`, `docs/sky-cleanup.md`, `docs/unity-viewer.md`, route guides, portable example config, launcher and ignore rules.

**Interfaces:** Documents executable commands from Tasks 1/2; training must work without requesting a Unity project.

- [ ] Add regression checks for config-relative tool paths and optional Unity publication. Expected: old behavior fails.
- [ ] Make executable discovery configurable; record tested GPU dependencies. Preserve old README as historical section documentation.
- [ ] Write the new README and detailed usage guides; include small result images and clear current limitations.
- [ ] Run full CPU suite, CLI help/dry runs and new-checkout smoke verification. Expected: all pass without QuestSBTC.
- [ ] Commit documentation and portability changes.

### Task 4: Review and publish

**Files:** Final release verification notes and any review fixes.

**Interfaces:** Consumes all preceding changes; produces a verified origin/master commit.

- [ ] Run final code review, address material findings, verify no datasets/secrets/game assets staged.
- [ ] Fetch origin, fast-forward master, push normally and verify remote SHA. Expected: local and remote master match and Track3DGS is clean.
