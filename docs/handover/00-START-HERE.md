# Track3DGS — Project Handover

**Read this first.** This folder is a complete handover of the Track3DGS project from
the development session (human owner + Claude Code) to whoever continues the work.
It is written to be read cold — you do not need to have seen the original session.

Last updated: **2026-09-28**. Status: **Phase 1 (reconstruction pipeline) is
production-proven** — 27 forest-track models delivered from ~833 m of Track01. The
Unity/VR runtime (LOD streaming + auto-drive) is **in early integration**.

---

## Read the docs in this order

| # | Document | What it gives you |
|---|----------|-------------------|
| 00 | **This file** | Orientation, the 2-minute summary, ground rules |
| 01 | [`01-project-overview.md`](01-project-overview.md) | What the project *is*: the goal, the two sub-projects (pipeline + Unity), the architecture, hardware, and current status |
| 02 | [`02-decisions-and-lessons.md`](02-decisions-and-lessons.md) | **The most valuable doc.** Every important decision, everything tried-and-rejected, and *why*. Read before changing anything — it stops you re-fighting settled battles |
| 03 | [`03-pipeline-steps-reference.md`](03-pipeline-steps-reference.md) | Every pipeline step: exact command, inputs, outputs, file formats, worked examples |
| 04 | [`04-file-structure.md`](04-file-structure.md) | Where everything lives: both repos, the `data/` layout, tools, Python environments, the precious calibration files |
| 05 | [`05-current-state-and-next-steps.md`](05-current-state-and-next-steps.md) | Exactly where work stopped, what is done vs pending, and the recommended roadmap |

Also read the repo-root [`README.md`](../../README.md). It is the terse operational
reference and is kept current; these handover docs add the depth, the history, and the
"why" that the README omits. The original design documents are in
[`docs/superpowers/`](../superpowers/) (the spec and the initial implementation plan) —
useful for original intent, but **superseded in many places** by what was actually
built (doc 02 tracks the divergences).

---

## The 2-minute summary

A 360° camera mounted on an armed vehicle recorded a **12 km forest track** (8K
equirectangular, 30 fps). Two things are being built:

1. **The reconstruction pipeline** (`C:\Work\Unity\DSTA\Track3DGS`, this repo) — turns
   that video, section by section, into **3D Gaussian Splatting (3DGS)** models. One
   PowerShell command runs a ~25–50 m section end-to-end; another runs a whole track
   unattended. This is done and proven.

2. **The Unity VR simulator** (`C:\Work\Unity\DSTA\QuestSBTC`, a *separate* Unity
   project) — an existing battle-tank crew simulator for **Meta Quest 3**. The splat
   sections are loaded into it and streamed with a distance-based LOD system so the
   crew "drives" through the reconstructed forest. This is early-stage.

The pipeline in one line:

```
cut video → extract sharp frames → mask (vehicle + sky) → 8 pinhole views per frame
→ COLMAP global SfM → auto-level → Splatfacto training → 3-tier artifact pruning
→ export PLY + quality report
```

**Track01 is fully reconstructed**: 27 sections, ~833 m, delivered to `data/Export/`.
The 26-section unattended batch ran **22.5 hours with zero failures**, every quality
gate green (mean masked PSNR ~20.5 dB). The owner is hand-polishing each model in
SuperSplat (results in `data/Export/Edited/`).

---

## Ground rules the owner worked by (inherit these)

1. **No shortcuts without asking.** The owner reviews each stage's visual result (a
   "smoke test") before the next stage runs. Automate everything, but when you build
   something new, gate on their visual verdict.
2. **Prefer existing tools** over custom UIs — LosslessCut (cutting), COLMAP GUI and
   SuperSplat (inspection). Custom code only where the pipeline genuinely needs it.
3. **Runtime budget matters as much as marginal quality.** A quality upgrade costing
   +45% wall-clock was rejected; a 2× speedup was adopted. Current quality
   (~20.5 dB masked PSNR) is the accepted standard — don't chase decimals at the cost
   of the 12 km timeline.
4. **Calibration files are precious.** Three files encode one-time human measurements
   (the vehicle mask, the sky prior, the mount key). Two of them live under `data/`,
   which is **not** in git. Back them up. Losing them means manual re-work. See doc 04.
5. **Manual polish is first-class.** The owner cleans models in SuperSplat; their edits
   can be recovered mathematically (attribute-fingerprint matching — doc 02), so
   hand-edited PLYs are legitimate deliverables, not throwaways.
6. **Everything is under test.** The Python pipeline has 58 passing unit tests
   (`cd pipeline; ..\.venv\Scripts\python.exe -m pytest`). Keep them green.

---

## Fastest way to prove the environment works

```powershell
cd C:\Work\Unity\DSTA\Track3DGS\pipeline
..\.venv\Scripts\python.exe -m pytest -q        # 58 tests, ~5 s, no GPU needed
```

If that passes, the utility environment is intact. The GPU/training environment
(`.venv-train` + `cuda_env.bat`) is the fragile one — doc 04 explains how it is built
and how to verify it. To run a real section end-to-end, see doc 03.
