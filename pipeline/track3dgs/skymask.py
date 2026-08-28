"""Per-frame sky masks via SegFormer (ADE20K, class 2 = sky).

Sky pixels are excluded from the training loss so the optimizer never
creates sky splats. Runs on the equirect frames; the views stage combines
each frame's sky mask with the static vehicle mask.

Requires the training venv (torch + transformers):
  cuda_env.bat ..\\.venv-train\\Scripts\\python.exe -m track3dgs.skymask --project ...
"""
import argparse

import cv2
import numpy as np

from .io_utils import Project, ensure_dir, read_jsonl

ADE20K_SKY_ID = 2


def sky_keep_mask(class_map, sky_ids=(ADE20K_SKY_ID,), dilate_px=8):
    """ADE20K class map (H,W int) -> keep-mask uint8 (255=keep, 0=sky).
    The sky region is dilated: eating a few leaf-edge pixels is harmless,
    leaking sky into training is not."""
    sky = np.isin(class_map, sky_ids).astype(np.uint8)
    if dilate_px:
        sky = cv2.dilate(sky, np.ones((dilate_px, dilate_px), np.uint8))
    return np.where(sky > 0, 0, 255).astype(np.uint8)


def bright_sky_mask(img_bgr, row_frac=0.45, min_val=235, max_sat=45):
    """Blown-out white sky (sun glare, bright clouds) sometimes escapes the
    segmentation class. Backstop: very bright + low-saturation pixels above
    the equirect horizon band are sky. Returns uint8 1=sky."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 2] >= min_val) & (hsv[:, :, 1] <= max_sat)
    bright[int(len(img_bgr) * row_frac):, :] = False
    return bright.astype(np.uint8)


def combine_keep_masks(a, b):
    """Logical AND of two keep-masks; b is resized to a's shape if needed."""
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    return np.minimum(a, b)


def run_skymask(project_dir, work_width=2048, dilate_px=8,
                model_name="nvidia/segformer-b2-finetuned-ade-512-512"):
    import torch
    from transformers import (SegformerForSemanticSegmentation,
                              SegformerImageProcessor)
    from tqdm import tqdm

    p = Project(project_dir)
    out_dir = ensure_dir(p.root / "sky_masks")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device).eval()

    recs = read_jsonl(p.frames_meta)
    sky_fracs = []
    for r in tqdm(recs, desc="skymask"):
        img = cv2.imread(str(p.frames_dir / r["name"]))
        h, w = img.shape[:2]
        small = cv2.resize(img, (work_width, work_width // 2))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits          # (1, 150, h/4, w/4)
        up = torch.nn.functional.interpolate(
            logits, size=small.shape[:2], mode="bilinear", align_corners=False)
        class_map = up.argmax(dim=1)[0].cpu().numpy()
        class_map[bright_sky_mask(small) > 0] = ADE20K_SKY_ID
        keep_small = sky_keep_mask(class_map, dilate_px=dilate_px)
        keep = cv2.resize(keep_small, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / (r["name"].rsplit(".", 1)[0] + ".png")), keep)
        sky_fracs.append(float((keep == 0).mean()))
    print(f"sky fraction: mean {100*np.mean(sky_fracs):.1f}% "
          f"min {100*np.min(sky_fracs):.1f}% max {100*np.max(sky_fracs):.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--work-width", type=int, default=2048)
    ap.add_argument("--dilate", type=int, default=8, dest="dilate_px")
    a = ap.parse_args()
    run_skymask(a.project, a.work_width, a.dilate_px)


if __name__ == "__main__":
    main()
