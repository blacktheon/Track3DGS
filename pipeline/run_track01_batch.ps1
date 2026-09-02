# Fully unattended Track01 batch: all remaining sections through steps 1-8.
# Continues on per-section failure; prints a summary table at the end.
# Section01 of track01-1 is already done (mount-key calibration section).
$ErrorActionPreference = "Continue"
$batch = @()
foreach ($i in 2..9)  { $batch += ,@("track01-1", $i) }
foreach ($i in 1..7)  { $batch += ,@("track01-2", $i) }
foreach ($i in 1..11) { $batch += ,@("track01-3", $i) }

$results = @()
foreach ($job in $batch) {
    $vid, $n = $job
    $sec = "section{0:d2}" -f $n
    $proj = "..\data\{0}_s{1:d2}" -f ($vid -replace "track", "t"), $n
    $export = "{0}-S{1}" -f ($vid -replace "track","Track"), $n
    Write-Host ("=" * 60)
    Write-Host ">>> BATCH: $vid $sec -> $proj (export: $export)"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        .\run_section.ps1 -Video "..\data\raw\$vid\$sec.mp4" -Project $proj `
            -SpeedKmh 10 -ExportName $export
        $status = "OK"
    } catch {
        $status = "FAILED: $($_.Exception.Message)"
    }
    $sw.Stop()
    $results += [pscustomobject]@{
        section = "$vid/$sec"; status = $status
        minutes = [math]::Round($sw.Elapsed.TotalMinutes, 1)
    }
    $results | ConvertTo-Csv -NoTypeInformation |
        Set-Content "..\data\batch_track01_status.csv" -Encoding utf8
}
Write-Host ("=" * 60)
$results | Format-Table -AutoSize
$ok = ($results | Where-Object status -eq "OK").Count
Write-Host "batch complete: $ok/$($results.Count) sections OK"
