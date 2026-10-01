@echo off
setlocal
rem Usage: cuda_env.bat command args...
rem Override TRACK3DGS_VCVARS64, TRACK3DGS_MSVC_VERSION and CUDA_PATH if needed.
rem CUDA 12.8 needs the v143/14.4x toolset, not newer 14.5x headers.
if not defined TRACK3DGS_MSVC_VERSION set "TRACK3DGS_MSVC_VERSION=14.4"
if defined TRACK3DGS_VCVARS64 goto found_vcvars
set "TRACK3DGS_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%TRACK3DGS_VSWHERE%" goto missing_vcvars
for /f "usebackq delims=" %%i in (`"%TRACK3DGS_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "TRACK3DGS_VSROOT=%%i"
if not defined TRACK3DGS_VSROOT goto missing_vcvars
set "TRACK3DGS_VCVARS64=%TRACK3DGS_VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
:found_vcvars
if not exist "%TRACK3DGS_VCVARS64%" goto missing_vcvars
call "%TRACK3DGS_VCVARS64%" -vcvars_ver=%TRACK3DGS_MSVC_VERSION% >nul
if errorlevel 1 exit /b 1
if not defined CUDA_PATH set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
  echo CUDA toolkit not found. Install CUDA 12.8 and set CUDA_PATH. 1>&2
  exit /b 1
)
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%~dp0..\.venv-train\Scripts;%PATH%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
rem our own checkpoints are trusted; torch>=2.6 weights_only default breaks ns-export
set "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
%*
exit /b %errorlevel%
:missing_vcvars
echo MSVC build tools not found. Install v143 C++ tools or set TRACK3DGS_VCVARS64. 1>&2
exit /b 1
