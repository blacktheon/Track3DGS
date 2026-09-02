# Track01 section boundaries (user-curated 2026-09-02, videos re-exported
# body-locked from .insv). Lossless keyframe cuts: -ss before -i snaps the
# start to the previous keyframe. Exports carry integer-second keyframes
# (verified: snap loss = 0), and sections already overlap ~2 s by design,
# so durations are cut exactly to the requested span - no extra buffer.
$src = "C:\Users\black\OneDrive\Desktop\BattleTank360Videos\Track01"
$dst = "C:\Work\Unity\DSTA\Track3DGS\data\raw"

$cuts = @{
    "Track01-1" = @(
        @(0, 9), @(7, 16), @(14, 26), @(24, 35), @(33, 49),
        @(47, 57), @(55, 65), @(63, 71), @(69, $null)
    )
    "Track01-2" = @(
        @(0, 7), @(5, 20), @(18, 32), @(30, 40), @(38, 50),
        @(48, 60), @(58, $null)
    )
    "Track01-3" = @(
        @(0, 9), @(7, 19), @(17, 29), @(27, 37), @(35, 50),
        @(48, 60), @(58, 72), @(70, 81), @(79, 90), @(88, 100), @(98, $null)
    )
}

foreach ($video in $cuts.Keys | Sort-Object) {
    $outDir = Join-Path $dst ($video.ToLower())
    New-Item -ItemType Directory -Force $outDir | Out-Null
    $sections = $cuts[$video]
    for ($i = 0; $i -lt $sections.Count; $i++) {
        $s = $sections[$i]
        $name = "section{0:d2}.mp4" -f ($i + 1)
        $out = Join-Path $outDir $name
        $args = @("-y", "-hide_banner", "-loglevel", "error",
                  "-ss", "$($s[0])", "-i", (Join-Path $src "$video.mp4"))
        if ($null -ne $s[1]) { $args += @("-t", "$($s[1] - $s[0])") }
        $args += @("-c", "copy", "-an", "-avoid_negative_ts", "make_zero", $out)
        & ffmpeg @args
        if ($LASTEXITCODE -ne 0) { throw "cut failed: $video $name" }
        $d = ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $out
        "{0}\{1}: {2:N1} s" -f $video.ToLower(), $name, [double]$d
    }
}
"all cuts complete"
