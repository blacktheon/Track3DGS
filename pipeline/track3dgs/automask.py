"""Auto-generate the vehicle-body mask: pixels static across all frames are
vehicle/rig (camera is rigidly mounted), moving pixels are world.
Writes mask_equirect.png (white=keep) + mask_overlay.jpg for visual QC."""
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from .io_utils import Project, read_jsonl


def compute_static_mask(frames, var_thresh=5.0, protect_top_frac=0.0):
    """frames: list of HxWx3 uint8. Returns HxW uint8 mask, 0=static(vehicle).
    Rows above protect_top_frac*H are always kept (sky is static but not vehicle)."""
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                      for f in frames])
    std = stack.std(axis=0)
    mask = np.where(std < var_thresh, 0, 255).astype(np.uint8)
    if protect_top_frac > 0:
        mask[:int(len(mask) * protect_top_frac), :] = 255
    mask[int(len(mask) * 0.85):, :] = 0   # bottom band is always hull on a roof mount
    # fill holes in the vehicle blob (reflective hull parts vary frame-to-frame),
    # then drop small false-positive islands in the world region
    k_close = np.ones((21, 21), np.uint8)
    k_open = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def run_automask(project_dir, work_width=960, var_thresh=12.0, max_frames=60,
                 dilate_px=8, protect_top_frac=0.38, force=False):
    p = Project(project_dir)
    if p.mask_path.exists() and not force:
        raise SystemExit(f"{p.mask_path} already exists (possibly hand-painted); "
                         "re-run with --force to overwrite it")
    recs = read_jsonl(p.frames_meta)
    step = max(1, len(recs) // max_frames)
    sample = recs[::step]

    first = cv2.imread(str(p.frames_dir / recs[0]["name"]))
    full_h, full_w = first.shape[:2]
    work_h = int(full_h * work_width / full_w)

    frames = []
    for r in tqdm(sample, desc="automask"):
        img = cv2.imread(str(p.frames_dir / r["name"]))
        frames.append(cv2.resize(img, (work_width, work_h)))

    mask_small = compute_static_mask(frames, var_thresh, protect_top_frac)
    if dilate_px:  # erode the keep-region: safety margin around the hull
        mask_small = cv2.erode(mask_small, np.ones((dilate_px, dilate_px), np.uint8))
    mask = cv2.resize(mask_small, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(p.mask_path), mask)

    # QC overlay: masked-out area tinted red on the reference frame
    overlay = first.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.4 +
                          np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    cv2.imwrite(str(p.root / "mask_overlay.jpg"), overlay)
    kept = float((mask == 255).mean()) * 100
    print(f"mask: {kept:.1f}% of frame kept, overlay -> mask_overlay.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--work-width", type=int, default=960)
    ap.add_argument("--var-thresh", type=float, default=12.0)
    ap.add_argument("--dilate", type=int, default=8, dest="dilate_px")
    ap.add_argument("--protect-top", type=float, default=0.38)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_automask(a.project, a.work_width, a.var_thresh, dilate_px=a.dilate_px,
                 protect_top_frac=a.protect_top, force=a.force)


if __name__ == "__main__":
    main()
