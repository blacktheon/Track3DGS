# 04 — File Structure, Environments & Tools

## 1. The two project roots

| Path | What | In git? |
|------|------|---------|
| `C:\Work\Unity\DSTA\Track3DGS` | The reconstruction pipeline (this repo) | Yes |
| `C:\Work\Unity\DSTA\QuestSBTC` | The Unity VR tank simulator (separate project) | Separate repo/none |

Plus external tools under `C:\Work\tools\` (§4).

## 2. Pipeline repo layout (`C:\Work\Unity\DSTA\Track3DGS`)

```
Track3DGS/
  README.md                    operational quick-reference (kept current)
  .gitignore                   ignores data/, .venv*, __pycache__, *.ply, *.msg
  .venv/                       utility Python env (NOT in git)
  .venv-train/                 GPU/training Python env (NOT in git)
  docs/
    handover/                  <- you are here (00..05)
    superpowers/
      specs/2026-08-06-track3dgs-pipeline-design.md      original design spec
      plans/2026-08-06-phase1-offline-pipeline.md        original implementation plan
  pipeline/
    track3dgs/                 the Python package (one module per stage)
      io_utils.py              Project layout + json/jsonl helpers (READ THIS FIRST)
      extract.py  views.py  skymask.py  automask.py       stages 1-3 + views
      track.py  trajectory.py  colmap_export.py           stage 4 (SfM) + pose math
      level.py                 stage 5 (mount-calibration levelling)
      cells.py  train.py  skyprune.py  slice.py  pack.py  stages 6-7 + tiling
      qc.py  evalpsnr.py        quality report + masked-PSNR tool
      gstrain.py               EXPERIMENTAL custom trainer (lost the A/B; not used)
    tests/                     58 pytest tests, one per module (no GPU needed)
    requirements.txt           utility-env deps
    pytest.ini
    cuda_env.bat               wrapper: MSVC v143 + CUDA 12.8 + UTF-8 + venv PATH
    run_section.ps1            one section, steps 1-9, timed
    run_track01_batch.ps1      whole track, unattended, resumable
    cuts_track01.ps1           section cut timetable -> lossless cuts
    mount_calibration.json     *** PRECIOUS: the mount key (body-locked) ***
    mount_calibration_stabilized_backup.json   old key (pre-body-locked)
    maskcheck_track01.py  skycompare_track01.py  continue_section02.ps1   one-off tools
  data/                        *** NOT in git — see §3 ***
```

Key reading order for a new dev: `io_utils.py` (the file contract) → `track.py` and
`trajectory.py` (the trickiest math) → `run_section.ps1` (how it all chains).

## 3. The `data/` folder (NOT in git — back it up separately)

```
data/
  raw/                                  source video + calibration inputs
    track01-1/section01.mp4 ... section09.mp4     Track01 video 1, cut into sections
    track01-2/section01.mp4 ... section07.mp4     video 2
    track01-3/section01.mp4 ... section11.mp4     video 3
    track01-1i/                          inverse (180deg) cuts (experiment, unused)
    track01_vehicle_mask.png   *** PRECIOUS: hand-painted vehicle mask ***
    track01_sky_prior.png      *** PRECIOUS: hand-drawn sky-candidate zone ***
    track01_vehicle_mask_inverse.png     rolled variants for inverse export
    track01_sky_prior_inverse.png
    track01_maskcheck/  track01_skycompare/        QC image galleries
    section01.mp4 section02.mp4 section03.mp4       older Track00 test sections
  t01-1_s02/ ... t01-3_s11/             per-section project dirs (26 batch sections)
  section01/ section02/ section03/      older Track00 sections (section01 has tiles/)
  Export/                               *** THE DELIVERABLES ***
    Track01-1-S2.ply ... Track01-3-S11.ply    pipeline output, one per section (~200 MB)
    Track01-*-S*.ssproj                        owner's SuperSplat polish projects
    Edited/                                    owner's hand-POLISHED final PLYs
      Track01-1-S1.ply ... Track01-2-S6.ply    (in progress, ~15 done)
  batch_track01_status.csv              batch run log
```

**Important nuances:**
- `data/Export/*.ply` are the **raw pipeline deliverables** (whole-section models).
- `data/Export/Edited/*.ply` are the owner's **hand-polished** versions (cleaned in
  SuperSplat). These are the *preferred* deliverables where they exist. As of handover
  ~15 of 27 are polished.
- `Track01-1-S1` has no raw `.ply` in `Export/` (only `.ssproj`) because S1 was the
  mount-calibration section and its polished version lives in `Edited/`.
- Per-section dirs (`t01-1_sNN`) can be regenerated from `raw/` by re-running; they are
  large (frames+views+colmap). Safe to delete to reclaim space if `Export/` is backed
  up, but you lose the ability to re-`slice`/`pack` without a full re-run.

## 4. External tools (`C:\Work\tools\`)

| Tool | Path | Purpose |
|------|------|---------|
| **COLMAP 4.1.1** (CUDA build) | `C:\Work\tools\colmap\` | SfM. Run the GUI via `COLMAP.bat`, **not** `bin\colmap.exe` (the .bat sets Qt plugin paths). Pipeline calls `bin\colmap.exe` directly for headless. |
| **LosslessCut** | `C:\Work\tools\LosslessCut\` | Manual lossless video cutting (keyframe stream-copy). |
| **SuperSplat** | web: superspl.at/editor | Splat viewing, levelling, manual polish. Runs locally in-browser. |
| **VS Build Tools + v143 toolset** | system | MSVC 14.4x — required to JIT-compile gsplat (CUDA 12.8 needs the VS2022 toolset, not VS2026). |
| **CUDA Toolkit 12.8** | system | nvcc for gsplat kernels. |

## 5. Python environments — the fragile part

Two venvs, deliberately separate (nerfstudio pins conflict with the utility deps).

### `.venv` — utility (no GPU)
Python 3.14. `pip install -r pipeline/requirements.txt`. Runs extract, views, track,
level, cells, skyprune, slice, pack, and all tests. Fast to rebuild.

### `.venv-train` — GPU/training (the hard one)
Python **3.11** (nerfstudio doesn't support 3.14). Built roughly as:
```
torch==2.7.1 torchvision==0.22.1  (--index-url .../cu128)   # Blackwell sm_120
fpsample==0.3.3                    # pinned: 1.0.x has no Windows wheel
nerfstudio==1.1.5
gsplat==1.4.0                      # JIT-compiles CUDA kernels on first use
transformers                       # Mask2Former sky segmentation
```
gsplat's kernels compile on first run and **need MSVC v143 + CUDA 12.8 on PATH** — that
is exactly what `cuda_env.bat` provides. Always launch the train python through it:
```powershell
.\cuda_env.bat ..\.venv-train\Scripts\python.exe -m track3dgs.train ...
```

**A patch lives inside this venv, not in git:** `nerfstudio/.../full_images_datamanager.py`
was edited to skip `pin_memory` for >1000-view sections (else CUDA OOM). **Reapply it
after any nerfstudio reinstall** — symptom is an out-of-memory crash right after
"caching images" on large sections (doc 02 §H3).

### Verifying the environments
```powershell
cd pipeline
..\.venv\Scripts\python.exe -m pytest -q                    # 58 pass, utility env OK
.\cuda_env.bat ..\.venv-train\Scripts\python.exe -c "import torch,gsplat,nerfstudio; print(torch.cuda.get_device_name(0))"
```

## 6. The precious calibration files (one-time human work)

These encode manual measurements. Losing them = redoing that manual work. Two of the
three are under `data/` (not in git) — **back them up.**

| File | What it is | If lost |
|------|-----------|---------|
| `data/raw/track01_vehicle_mask.png` | hand-painted hull mask (white=keep) | repaint on a reference frame (~30 min) |
| `data/raw/track01_sky_prior.png` | hand-drawn sky-candidate zone | redraw (~15 min) |
| `pipeline/mount_calibration.json` | the mount key (camera axes in a level world) — in git | re-measure: run 1 section, level it in SuperSplat, recover via fingerprint-Kabsch |

All three are **per-camera-mount / per-export-style**. They are valid for the current
body-locked Track01 footage. New footage with a remounted camera or a different export
needs them re-derived (doc 02 §D3, §F1).

## 7. Unity project layout (`C:\Work\Unity\DSTA\QuestSBTC`)

A large pre-existing tank-sim project. What's relevant to this work:
```
Assets/
  Scripts/
    GSObject.cs            marks a splat section for LOD (add next to GaussianSplatRenderer)
    GSLODManager.cs        enables only the closest N GSObjects to a tracking target
    AutoDriveWaypoints.cs  waypoint auto-drive of the ControlRoom (T=teleport, F=play/pause)
    (~40 other pre-existing sim scripts: cameras, controls, enemy AI, etc.)
  Settings/
    PC_Renderer.asset          URP renderer WITH GaussianSplatURPFeature (VR eyes)
    Instruments_NoSplat.asset  URP renderer WITHOUT it (periscope cameras) - doc 02 I2
  (imported GaussianSplatAsset objects created from the Export PLYs)
```
Scene hierarchy of note: `ControlRoom` (the crew platform + Main Camera + controls),
`Track01` (parent of the loaded splat sections), `GSLODManager`, `Waypoints`.
The splat renderer is the **UnityGaussianSplatting** package (aras-p), added via git URL.
