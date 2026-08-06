"""Stage 2: equirect frames -> perspective crops (+ per-view masks + intrinsics)."""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import py360convert
from tqdm import tqdm

from .io_utils import Project, ensure_dir, read_jsonl, write_json

DEFAULT_YAWS = [-135, -90, -45, 45, 90, 135]


def intrinsics(fov_deg, size):
    f = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return {"width": size, "height": size, "fov_deg": fov_deg,
            "fx": f, "fy": f, "cx": size / 2.0, "cy": size / 2.0}


def view_name(frame_name, yaw):
    stem = Path(frame_name).stem
    return f"{stem}_y{yaw:+04d}.jpg"


def _e2p(img, yaw, fov, size):
    return py360convert.e2p(img, fov_deg=(fov, fov), u_deg=yaw, v_deg=0,
                            out_hw=(size, size))


def run_views(project_dir, yaws=DEFAULT_YAWS, fov=100.0, size=1600):
    p = Project(project_dir)
    ensure_dir(p.views_dir)
    ensure_dir(p.views_masks_dir)
    frames = read_jsonl(p.frames_meta)

    mask = cv2.imread(str(p.mask_path), cv2.IMREAD_GRAYSCALE)
    mask_crops = {}
    for yaw in yaws:
        mc = _e2p(np.stack([mask] * 3, axis=-1), yaw, fov, size)[:, :, 0]
        mask_crops[yaw] = (mc > 127).astype(np.uint8) * 255

    for rec in tqdm(frames, desc="views"):
        img = cv2.imread(str(p.frames_dir / rec["name"]))
        for yaw in yaws:
            name = view_name(rec["name"], yaw)
            cv2.imwrite(str(p.views_dir / name), _e2p(img, yaw, fov, size),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(p.views_masks_dir / Path(name).with_suffix(".png").name),
                        mask_crops[yaw])

    write_json(p.views_meta, intrinsics(fov, size) | {"yaws": list(yaws)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--yaws", default=",".join(str(y) for y in DEFAULT_YAWS))
    ap.add_argument("--fov", type=float, default=100.0)
    ap.add_argument("--size", type=int, default=1600)
    a = ap.parse_args()
    run_views(a.project, [int(y) for y in a.yaws.split(",")], a.fov, a.size)


if __name__ == "__main__":
    main()
