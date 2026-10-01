# Installation

The supported training path is **Windows x64 with an NVIDIA CUDA GPU**. The
repository and included Unity viewer work independently of QuestSBTC. Use a short
checkout path such as `C:\Work\Track3DGS`: deeply nested paths can exceed Windows
path limits inside Unity PackageCache importers. Capture
videos, masks and trained PLYs are local inputs; none are downloaded by cloning.

## Utility environment

Install Python 3.11 or newer and FFmpeg (`ffmpeg` and `ffprobe` on `PATH`). The
CPU tests exercise FFmpeg as well as Python code. Python 3.11 is a convenient
choice for both environments; the development utility environment also uses 3.14.

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r pipeline\requirements.txt
Push-Location pipeline
..\.venv\Scripts\python.exe -m pytest -q
Pop-Location
```

## Reconstruction and GPU environment

Install COLMAP **4.1.1** with its CUDA dependencies. Its `colmap.exe` must be on
`PATH`, or set `toolchain.colmap` in the route configuration to the executable.
The route rig uses COLMAP's 4.1 panorama-rig workflow; older builds are unsuitable.
Install CUDA Toolkit **12.8**, a compatible NVIDIA driver, and Visual Studio C++
Build Tools with the **v143 / 14.4x** toolset. Newer 14.5x headers are not the tested
combination. The toolkit is needed for gsplat's first-run JIT compilation.

Create a separate Python **3.11** environment:

```powershell
py -3.11 -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-train\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv-train\Scripts\python.exe -m pip install -r pipeline\requirements-gpu.txt
Push-Location pipeline
.\cuda_env.bat ..\.venv-train\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0))"
Pop-Location
```

Nerfstudio brings a large dependency tree. `fpsample==0.3.3` is pinned for the
Windows environment; use the Python 3.11 Windows wheel where available, or build
it with the same C++ toolchain. The [recorded environment](validation/gpu-environment-2026-10-01.json)
lists the versions present during verification. It is an audit snapshot, not a
promise that every unrelated installed package is necessary. The GPU requirements
file pins the important training interfaces and passed a fresh dependency-resolution
dry run; a complete fresh GPU installation
is distinct from the verified existing-environment rendering tests.

`cuda_env.bat` discovers Visual Studio through `vswhere` and respects `CUDA_PATH`.
For a nonstandard installation, set these before running it:

```powershell
$env:TRACK3DGS_VCVARS64 = 'D:\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$env:TRACK3DGS_MSVC_VERSION = '14.4'
$env:CUDA_PATH = 'D:\CUDA\v12.8'
```

Route training installs its bounded image-cache adapter **in the subprocess**.
It does not require the historical hand-edited Nerfstudio `pin_memory` patch.
The adapter deliberately requires Nerfstudio 1.1.5. Only load checkpoints that
you trust; the training launcher enables full checkpoint deserialization.

## Inputs for a new capture

Copy `pipeline/configs/route.example.json` to a new configuration. Paths in the
JSON, including tool paths containing a slash, resolve relative to that JSON.
Bare tool names such as `colmap` are resolved from `PATH`.

Provide:

- A continuous equirectangular 360° video. Body-locked exports work with a fixed
  vehicle mask; stabilization that moves the vehicle in the panorama needs a
  different masking strategy.
- A grayscale vehicle **keep mask**, with 0 on the vehicle and 255 on usable
  surroundings. Match the video panorama dimensions.
- A grayscale sky prior for the union sky segmenter: 0 where sky is possible,
  255 elsewhere. This is a capture-specific prior, not the final per-frame sky
  segmentation. Inspect it against several frames before a long run.
- An approximate speed or another scale setting, and a new output revision.
  The bundled 10 km/h example is an assumption, not a general calibration.

The older `automask` command can propose a vehicle mask; inspect and correct it.
Do not reuse the Track01 mount calibration on another rig without verification.
The route example does not require that historical calibration.

The sky segmenter fetches its pretrained model on first use. Unity Package
Manager likewise downloads the pinned public URP dependencies on first open.

## Hardware and storage

The six-region Track02 pass and cleanup reproductions used an RTX 5060 Ti with
16 GB VRAM. That is a tested machine, not a guaranteed minimum specification.
Start with a short capture and one region. Original panoramas, eight pinhole
views per selected frame, masks and checkpoints can consume substantial disk
space. The default route image cache bounds decoded training images to 32;
`TRACK3DGS_IMAGE_CACHE` can lower that at the cost of more disk reads.

Training/QC jobs are sequential and share a Windows GPU lease. Unload large
Unity previews while training if they compete for memory. For visual-only
inspection, install Unity as described in the [viewer guide](unity-viewer.md).
