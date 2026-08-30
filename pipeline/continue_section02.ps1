# One-off continuation of the timed section02 run (steps 1-4a harvested from log).
$ErrorActionPreference = "Stop"
$py = "..\.venv\Scripts\python.exe"
$pyt = "..\.venv-train\Scripts\python.exe"
$Project = "..\data\section02"
$timing = @(
    [pscustomobject]@{ step = "1-extract"; minutes = 1.0 },
    [pscustomobject]@{ step = "2-vehicle-mask"; minutes = 0.0 },
    [pscustomobject]@{ step = "3-skymask"; minutes = 1.2 },
    [pscustomobject]@{ step = "4a-views"; minutes = 4.0 }
)
function Step($name, $block) {
    Write-Host ">>> $name"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $block
    if ($LASTEXITCODE -ne 0) { throw "step '$name' failed with exit $LASTEXITCODE" }
    $sw.Stop()
    $script:timing += [pscustomobject]@{ step = $name; minutes = [math]::Round($sw.Elapsed.TotalMinutes, 2) }
    Write-Host ">>> $name done in $([math]::Round($sw.Elapsed.TotalMinutes,1)) min"
}
if (Test-Path "$Project\track\colmap_work") { Remove-Item "$Project\track\colmap_work" -Recurse -Force }
Step "4b-track"   { & $py -m track3dgs.track --project $Project --speed-kmh 10 --overlap 48 --mapper glomap }
Step "5-level"    { & $py -m track3dgs.level --project $Project }
Step "6a-cells"   { & $py -m track3dgs.cells --project $Project }
Step "6b-train"   { .\cuda_env.bat $pyt -m track3dgs.train --project $Project --cell 0 }
Step "7-skyprune" { & $py -m track3dgs.skyprune --project $Project --cell 0 }
$timing += [pscustomobject]@{ step = "TOTAL"; minutes = [math]::Round(($timing | Measure-Object minutes -Sum).Sum, 1) }
$timing | Format-Table -AutoSize
$timing | ConvertTo-Csv -NoTypeInformation | Set-Content "$Project\timing.csv" -Encoding utf8
Write-Host "timing written"
