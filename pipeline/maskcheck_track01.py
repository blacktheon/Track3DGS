import cv2, numpy as np, subprocess, os
from pathlib import Path

raw = Path(r"..\data\raw")
out = raw / "track01_maskcheck"
out.mkdir(exist_ok=True)
picks = [("track01-1", "section01"), ("track01-1", "section05"), ("track01-1", "section09"),
         ("track01-2", "section01"), ("track01-2", "section04"), ("track01-2", "section07"),
         ("track01-3", "section01"), ("track01-3", "section06"), ("track01-3", "section11")]

mask = cv2.imread(r"..\data\section01\mask_equirect.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (1920, 960), interpolation=cv2.INTER_NEAREST)

frames = []
for vid, sec in picks:
    src = raw / vid / f"{sec}.mp4"
    tmp = out / f"ref_{vid}_{sec}.jpg"
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of",
                                "default=noprint_wrappers=1:nokey=1", str(src)],
                               capture_output=True, text=True).stdout)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(dur / 2),
                    "-i", str(src), "-vframes", "1", str(tmp)], check=True)
    img = cv2.resize(cv2.imread(str(tmp)), (1920, 960))
    frames.append(img)
    ov = img.copy()
    sel = mask <= 127
    ov[sel] = (ov[sel] * 0.35 + np.array([0, 0, 255]) * 0.65).astype(np.uint8)
    cv2.putText(ov, f"{vid} {sec}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                (255, 255, 255), 4)
    cv2.imwrite(str(out / f"maskfit_{vid}_{sec}.jpg"), ov)

stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                  for f in frames])
std = stack.std(axis=0)
static = (std < 12).astype(np.uint8)

# agreement between measured-static region and the existing mask (vehicle=black)
vehicle = (mask <= 127)
hull_zone = np.zeros_like(vehicle); hull_zone[300:, :] = True   # below sky band
static_v = static.astype(bool) & hull_zone
covered = float((static_v & vehicle).sum() / max(static_v.sum(), 1))
leaks = static_v & ~vehicle
print(f"static (vehicle) pixels covered by existing mask: {100*covered:.1f}%")
print(f"static pixels NOT covered (potential leaks): {int(leaks.sum()):,}")

heat = cv2.applyColorMap((np.clip(std, 0, 60) / 60 * 255).astype(np.uint8),
                         cv2.COLORMAP_TURBO)
edges = cv2.Canny(vehicle.astype(np.uint8) * 255, 50, 150)
heat[edges > 0] = (255, 255, 255)
cv2.putText(heat, "variance across 9 frames (dark=static) + mask outline",
            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
cv2.imwrite(str(out / "variance_vs_mask.jpg"), heat)
print("artifacts in", out)
