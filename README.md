# Track3DGS

Reconstruction of a 12 km forest track from 360° vehicle-mounted video as 3D Gaussian
Splatting models, streamed as 10 m tiles in Unity on Quest 3.

- Design spec: `docs/superpowers/specs/2026-08-06-track3dgs-pipeline-design.md`
- Implementation plan: `docs/superpowers/plans/2026-08-06-phase1-offline-pipeline.md`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r pipeline\requirements.txt
```

FFmpeg must be on PATH. Section cutting uses LosslessCut (`C:\Work\tools\LosslessCut`).

## Pipeline stages (run from `pipeline/`)

```powershell
python -m track3dgs.extract --video ..\data\raw\section01.mp4 --out ..\data\section01
python -m track3dgs.views   --project ..\data\section01
python -m track3dgs.track   --project ..\data\section01 --speed-kmh <V>
python -m track3dgs.cells   --project ..\data\section01
python -m track3dgs.train   --project ..\data\section01 --cell 0
python -m track3dgs.slice   --project ..\data\section01 --cell 0
python -m track3dgs.pack    --project ..\data\section01 --merge
```

Tests: `python -m pytest` from `pipeline/`.
