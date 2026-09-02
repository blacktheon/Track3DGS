"""A/B: user's prior+color sky algorithm vs current Mask2Former pipeline.

New algorithm: hand-drawn spatial prior (sky_mask.png, black = sky-possible
zone) AND per-pixel blue-to-white colour test. No segmentation model.

Samples 100 random frames across the three Track01 videos, writes stacked
side-by-side overlays + aggregate stats. Run under cuda_env + .venv-train.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np

from track3dgs.skymask import (ADE20K_SKY_ID, _class_map, _load_model,
                               bright_sky_mask, fill_enclosed_sky,
                               sky_color_test, sky_keep_mask)

RAW = Path(r"..\data\raw")
SRC = Path(r"C:\Users\black\OneDrive\Desktop\BattleTank360Videos\Track01")
OUT = RAW / "track01_skycompare"
W, H = 1920, 960


color_test = sky_color_test   # canonical tuned rule from the pipeline


def main():
    OUT.mkdir(exist_ok=True)
    prior = cv2.imread(str(RAW / "track01_maskcheck" / "sky_mask.png"),
                       cv2.IMREAD_GRAYSCALE)
    prior = cv2.resize(prior, (W, H), interpolation=cv2.INTER_NEAREST) <= 127

    durs = {"Track01-1": 77.3, "Track01-2": 66.0, "Track01-3": 110.8}
    rng = np.random.default_rng(42)
    total = sum(durs.values())
    samples = []
    for vid, d in durs.items():
        n = int(round(100 * d / total))
        samples += [(vid, float(t)) for t in rng.uniform(0.5, d - 0.5, n)]
    samples = samples[:100]

    import time
    proc, model = _load_model("mask2former", "cuda")
    frac_new, frac_cur, frac_uni = [], [], []
    t_m2f = t_color = 0.0
    for idx, (vid, t) in enumerate(samples):
        tmp = OUT / "_tmp.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}",
                        "-i", str(SRC / f"{vid}.mp4"), "-vframes", "1",
                        str(tmp)], check=True)
        img = cv2.resize(cv2.imread(str(tmp)), (W, H))

        t0 = time.perf_counter()
        sky_new = fill_enclosed_sky(color_test(img) & prior, prior)
        t_color += time.perf_counter() - t0

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_map, _ = _class_map("mask2former", proc, model, rgb, (H, W), "cuda")
        class_map[bright_sky_mask(img) > 0] = ADE20K_SKY_ID
        sky_cur = sky_keep_mask(class_map) == 0
        t_m2f += time.perf_counter() - t0

        sky_uni = sky_new | sky_cur
        frac_new.append(sky_new.mean())
        frac_cur.append(sky_cur.mean())
        frac_uni.append(sky_uni.mean())

        panels = []
        for sky, label in ((sky_cur, "current: mask2former"),
                           (sky_new, "new: prior + blue-to-white"),
                           (sky_uni, "UNION of both")):
            ov = img.copy()
            ov[sky] = (ov[sky] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
            cv2.putText(ov, f"{label}   [{vid} @ {t:.1f}s]", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            panels.append(ov)
        cv2.imwrite(str(OUT / f"cmp_{idx:03d}_{vid}_{t:05.1f}.jpg"),
                    np.vstack(panels))
        if idx % 20 == 0:
            print(f"{idx + 1}/100 done", flush=True)

    tmp.unlink(missing_ok=True)
    print(f"sky fraction: mask2former {100 * np.mean(frac_cur):.1f}%  "
          f"prior+color {100 * np.mean(frac_new):.1f}%  "
          f"UNION {100 * np.mean(frac_uni):.1f}%")
    print(f"per-frame cost: mask2former {1000 * t_m2f / 100:.0f} ms, "
          f"prior+color {1000 * t_color / 100:.1f} ms  "
          f"-> union adds {100 * t_color / t_m2f:.2f}% over mask2former alone")
    print(f"100 comparisons in {OUT}")


if __name__ == "__main__":
    main()
