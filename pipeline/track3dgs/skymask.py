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


def sky_keep_mask(class_map, sky_ids=(ADE20K_SKY_ID,), dilate_px=8,
                  min_dilate_area_frac=0.005):
    """ADE20K class map (H,W int) -> keep-mask uint8 (255=keep, 0=sky).
    Only large sky regions (the sky dome) are dilated for a safety margin;
    small sky gaps inside the canopy are left as-is so dilation does not
    eat the surrounding leaves."""
    sky = np.isin(class_map, sky_ids).astype(np.uint8)
    if dilate_px:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(sky)
        big = np.zeros_like(sky)
        thr = min_dilate_area_frac * sky.size
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= thr:
                big[labels == i] = 1
        big = cv2.dilate(big, np.ones((dilate_px, dilate_px), np.uint8))
        sky = np.maximum(sky, big)
    return np.where(sky > 0, 0, 255).astype(np.uint8)


def bright_sky_mask(img_bgr, row_frac=0.45, min_val=210, max_sat=60):
    """Blown-out white sky (sun glare, bright clouds) sometimes escapes the
    segmentation class. Backstop: very bright + low-saturation pixels above
    the equirect horizon band are sky. Returns uint8 1=sky."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 2] >= min_val) & (hsv[:, :, 1] <= max_sat)
    bright[int(len(img_bgr) * row_frac):, :] = False
    return bright.astype(np.uint8)


def sky_color_test(img_bgr, blue_hue=(95, 135), blue_min_sat=45,
                   blue_min_val=90, pale_min_val=205, pale_max_sat=40):
    """Blue-to-white sky colour test (bool per pixel), tuned to spare grey and
    green-ish pixels: the blue band starts past cyan-green (hue 95) and needs
    real saturation; 'pale' means near-clipping brightness AND a cold tone
    (B >= R) - warm grey rock and sunlit branches fail, sky whites pass."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = ((h >= blue_hue[0]) & (h <= blue_hue[1])
            & (s >= blue_min_sat) & (v >= blue_min_val))
    cold = img_bgr[:, :, 0].astype(np.int16) >= img_bgr[:, :, 2].astype(np.int16)
    pale = (v >= pale_min_val) & (s <= pale_max_sat) & cold
    return blue | pale


def fill_enclosed_sky(sky, prior_zone):
    """Inside the sky-candidate zone, non-sky islands NOT connected to the
    world (the area outside the zone) are enclosed by sky - i.e. clouds that
    failed the colour test - and get filled. Tree crowns poking into the zone
    stay: they connect to the world along the treeline."""
    inv = (~sky.astype(bool)) & prior_zone.astype(bool)
    n, lab = cv2.connectedComponents(inv.astype(np.uint8))
    world = (~prior_zone.astype(bool)).astype(np.uint8)
    near_world = cv2.dilate(world, np.ones((3, 3), np.uint8)).astype(bool)
    touching = np.unique(lab[near_world & inv])
    fill = inv & ~np.isin(lab, touching)
    return sky.astype(bool) | fill


def color_sky_mask(img_bgr, lat_cutoff_frac=0.52, **kwargs):
    """Latitude-cutoff variant of the colour rule (no hand-drawn prior).
    Returns uint8 1=sky."""
    sky = sky_color_test(img_bgr, **kwargs)
    sky[int(len(img_bgr) * lat_cutoff_frac):, :] = False
    return sky.astype(np.uint8)


def combine_keep_masks(a, b):
    """Logical AND of two keep-masks; b is resized to a's shape if needed."""
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    return np.minimum(a, b)


SEGFORMER = "nvidia/segformer-b2-finetuned-ade-512-512"
MASK2FORMER = "facebook/mask2former-swin-large-ade-semantic"


def _load_model(kind, device):
    if kind == "mask2former":
        from transformers import (AutoImageProcessor,
                                  Mask2FormerForUniversalSegmentation)
        proc = AutoImageProcessor.from_pretrained(MASK2FORMER)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(MASK2FORMER)
    else:
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)
        proc = SegformerImageProcessor.from_pretrained(SEGFORMER)
        model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER)
    return proc, model.to(device).eval()


def _class_map(kind, proc, model, rgb, out_hw, device):
    import torch
    inputs = proc(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    if kind == "mask2former":
        seg = proc.post_process_semantic_segmentation(
            out, target_sizes=[out_hw])[0]
        return seg.cpu().numpy(), None
    up = torch.nn.functional.interpolate(
        out.logits, size=out_hw, mode="bilinear", align_corners=False)
    probs = torch.softmax(up, dim=1)[0]
    return (up.argmax(dim=1)[0].cpu().numpy(),
            probs[ADE20K_SKY_ID].cpu().numpy())


DEFAULT_PRIOR = r"..\data\raw\track01_sky_prior.png"


def run_skymask(project_dir, work_width=2048, dilate_px=8, kind="mask2former",
                prior_path=DEFAULT_PRIOR):
    from tqdm import tqdm

    p = Project(project_dir)
    out_dir = ensure_dir(p.root / "sky_masks")
    proc = model = device = prior_small = None
    if kind in ("union", "prior"):
        prior = cv2.imread(str(prior_path), cv2.IMREAD_GRAYSCALE)
        if prior is None:
            raise SystemExit(f"sky prior not found: {prior_path}")
        prior_small = cv2.resize(prior, (work_width, work_width // 2),
                                 interpolation=cv2.INTER_NEAREST) <= 127
    if kind not in ("color", "prior"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc, model = _load_model("mask2former" if kind == "union" else kind,
                                  device)

    recs = read_jsonl(p.frames_meta)
    sky_fracs = []
    for r in tqdm(recs, desc=f"skymask[{kind}]"):
        img = cv2.imread(str(p.frames_dir / r["name"]))
        h, w = img.shape[:2]
        small = cv2.resize(img, (work_width, work_width // 2))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if kind == "color":
            class_map = np.full(small.shape[:2], 4, dtype=np.int64)  # 4 = tree
            class_map[color_sky_mask(small) > 0] = ADE20K_SKY_ID
            keep_small = sky_keep_mask(class_map, dilate_px=dilate_px)
        elif kind == "prior":
            sky = fill_enclosed_sky(sky_color_test(small) & prior_small,
                                    prior_small)
            keep_small = np.where(sky, 0, 255).astype(np.uint8)
        elif kind == "union":
            # mask2former path (semantic treeline) ...
            class_map, _ = _class_map("mask2former", proc, model, rgb,
                                      small.shape[:2], device)
            class_map[bright_sky_mask(small) > 0] = ADE20K_SKY_ID
            m2f_sky = sky_keep_mask(class_map, dilate_px=dilate_px) == 0
            # ... unioned with the prior+colour rule (canopy-gap precision)
            prior_sky = fill_enclosed_sky(sky_color_test(small) & prior_small,
                                          prior_small)
            keep_small = np.where(m2f_sky | prior_sky, 0, 255).astype(np.uint8)
        else:
            class_map, sky_prob = _class_map(kind, proc, model, rgb,
                                             small.shape[:2], device)
            if sky_prob is not None:
                # pale clouds are often only marginally "sky" for segformer
                class_map[sky_prob > 0.25] = ADE20K_SKY_ID
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
    ap.add_argument("--model", default="mask2former",
                    choices=["mask2former", "segformer", "color", "prior",
                             "union"])
    ap.add_argument("--prior", default=DEFAULT_PRIOR,
                    help="hand-drawn sky-candidate zone (black = sky possible)")
    a = ap.parse_args()
    run_skymask(a.project, a.work_width, a.dilate_px, a.model, a.prior)


if __name__ == "__main__":
    main()
