# Track3DGS: run one section through steps 1-7 with per-step timing.
# Usage: .\run_section.ps1 -Video ..\data\raw\section02.mp4 -Project ..\data\section02 `
#          -SpeedKmh 10 -Mapper glomap -MaskFrom ..\data\section01\mask_equirect.png
param(
    [Parameter(Mandatory)] [string]$Video,
    [Parameter(Mandatory)] [string]$Project,
    [Parameter(Mandatory)] [double]$SpeedKmh,
    [string]$Mapper = "glomap",
    [string]$MaskFrom = "..\data\raw\track01_vehicle_mask.png",
    [string]$SkyModel = "union",
    [string]$SkyPrior = "..\data\raw\track01_sky_prior.png",
    [int]$Overlap = 48
)
$ErrorActionPreference = "Stop"
$py = "..\.venv\Scripts\python.exe"
$pyt = "..\.venv-train\Scripts\python.exe"
$timing = @()

function Step($name, $block) {
    Write-Host ">>> $name"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $block
    if ($LASTEXITCODE -ne 0) { throw "step '$name' failed with exit $LASTEXITCODE" }
    $sw.Stop()
    $script:timing += [pscustomobject]@{ step = $name; minutes = [math]::Round($sw.Elapsed.TotalMinutes, 2) }
    Write-Host ">>> $name done in $([math]::Round($sw.Elapsed.TotalMinutes,1)) min"
}

Step "1-extract"  { & $py -m track3dgs.extract --video $Video --out $Project }
Step "2-vehicle-mask" { Copy-Item $MaskFrom (Join-Path $Project "mask_equirect.png") -Force; $global:LASTEXITCODE = 0 }
Step "3-skymask"  { .\cuda_env.bat $pyt -m track3dgs.skymask --project $Project --model $SkyModel --prior $SkyPrior }
Step "4a-views"   { & $py -m track3dgs.views --project $Project --yaws "-135,-90,-45,0,45,90,135,180" }
Step "4b-track"   { & $py -m track3dgs.track --project $Project --speed-kmh $SpeedKmh --overlap $Overlap --mapper $Mapper }
Step "5-level"    { & $py -m track3dgs.level --project $Project }
Step "6a-cells"   { & $py -m track3dgs.cells --project $Project }
Step "6b-train"   { .\cuda_env.bat $pyt -m track3dgs.train --project $Project --cell 0 }
Step "7-skyprune" { & $py -m track3dgs.skyprune --project $Project --cell 0 }

$timing += [pscustomobject]@{ step = "TOTAL"; minutes = [math]::Round(($timing | Measure-Object minutes -Sum).Sum, 1) }
$timing | Format-Table -AutoSize
$timing | ConvertTo-Csv -NoTypeInformation | Set-Content (Join-Path $Project "timing.csv") -Encoding utf8
Write-Host "timing written to $Project\timing.csv"
