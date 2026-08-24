"""Read/write COLMAP text models (cameras/images/points3D)."""
from pathlib import Path

import numpy as np

from .io_utils import ensure_dir
from .trajectory import world_to_camera


def write_colmap_model(out_dir, intr, images, points):
    out = ensure_dir(out_dir)

    (out / "cameras.txt").write_text(
        "# Camera list\n"
        f"1 PINHOLE {intr['width']} {intr['height']} "
        f"{intr['fx']} {intr['fy']} {intr['cx']} {intr['cy']}\n")

    lines = ["# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME"]
    for i, im in enumerate(images, start=1):
        qw, qx, qy, qz, tx, ty, tz = world_to_camera(im["T_wc"])
        lines.append(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {im['name']}")
        lines.append("")  # empty observations line
    (out / "images.txt").write_text("\n".join(lines) + "\n")

    plines = ["# 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]"]
    if points is not None:
        for pid, p in enumerate(np.asarray(points), start=1):
            x, y, z = p[:3]
            r, g, b = (int(p[3]), int(p[4]), int(p[5])) if len(p) >= 6 else (128, 128, 128)
            plines.append(f"{pid} {x} {y} {z} {r} {g} {b} 0.5")
    (out / "points3D.txt").write_text("\n".join(plines) + "\n")


def load_images_txt(path):
    """Parse an images.txt written by us or by COLMAP (2 lines per image,
    the second being the observations line, possibly empty)."""
    out = []
    lines = [l for l in Path(path).read_text().splitlines()
             if not l.startswith("#")]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        f = line.split()
        out.append({"id": int(f[0]),
                    "q": [float(v) for v in f[1:5]],
                    "t": [float(v) for v in f[5:8]],
                    "name": f[9]})
        i += 1  # skip the observations line (may be empty or long)
    return out


def load_points3d_txt(path):
    """Return (N,6) array: xyz + rgb. Ignores error/track fields."""
    pts = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split()
            pts.append([float(f[1]), float(f[2]), float(f[3]),
                        float(f[4]), float(f[5]), float(f[6])])
    return np.array(pts) if pts else None
