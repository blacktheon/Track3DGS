@echo off
rem Runs any command with MSVC + CUDA 12.8 on PATH (needed for gsplat JIT compile).
rem Usage: cuda_env.bat <command...>
rem Pin the VS2022 (v143 / 14.4x) toolset: CUDA 12.8's nvcc crashes on the
rem default VS2026 (14.5x) headers.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.4 >nul
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%~dp0..\.venv-train\Scripts;%PATH%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
%*
