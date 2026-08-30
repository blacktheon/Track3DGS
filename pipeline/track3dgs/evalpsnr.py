"""Objective quality check: render a splat PLY at real view poses and compare
against the actual crops (masked PSNR). Works for any PLY in the section's
global frame regardless of which mapper/trainer produced it.

Run inside cuda_env with the training venv (needs compiled gsplat).
"""
import argparse

import cv2
import numpy as np

from .colmap_export import load_images_txt
from .io_utils import Project, read_json
from .trajectory import quat_to_R


def load_splats(path):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"].data
    n = len(v)
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float32)))
    sh0 = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :]
    rest_names = sorted([m for m in v.dtype.names if m.startswith("f_rest_")],
                        key=lambda s: int(s.split("_")[-1]))
    if rest_names:
        rest = np.stack([v[m] for m in rest_names], 1).astype(np.float32)
        k = len(rest_names) // 3
        shN = rest.reshape(n, 3, k).transpose(0, 2, 1)     # Inria channel-major
    else:
        shN = np.zeros((n, 0, 3), np.float32)
    return means, scales, quats, op, sh0.astype(np.float32), shN


def run_eval(project_dir, ply_path, n_views=12, seed=0):
    import torch
    from gsplat import rasterization

    p = Project(project_dir)
    meta = read_json(p.views_meta)
    K = torch.tensor([[meta["fx"], 0, meta["cx"]],
                      [0, meta["fy"], meta["cy"]], [0, 0, 1.0]],
                     device="cuda", dtype=torch.float32)
    imgs = load_images_txt(p.colmap_dir / "images.txt")
    rng = np.random.default_rng(seed)
    sample = [imgs[i] for i in rng.choice(len(imgs), n_views, replace=False)]

    means, scales, quats, op, sh0, shN = load_splats(ply_path)
    dev = {k_: torch.tensor(x, device="cuda") for k_, x in
           [("m", means), ("s", scales), ("q", quats), ("o", op)]}
    colors = torch.tensor(np.concatenate([sh0, shN], axis=1), device="cuda")
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    psnrs = []
    for im in sample:
        qw, qx, qy, qz = im["q"]
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = im["t"]
        vm = torch.tensor(T, device="cuda")[None]
        W, H = meta["width"], meta["height"]
        with torch.no_grad():
            render, _, _ = rasterization(
                means=dev["m"], quats=dev["q"] / dev["q"].norm(dim=-1, keepdim=True),
                scales=dev["s"], opacities=dev["o"], colors=colors,
                sh_degree=sh_degree, viewmats=vm, Ks=K[None], width=W, height=H)
        pred = render[0, ..., :3].clamp(0, 1).cpu().numpy()
        gt = cv2.cvtColor(cv2.imread(str(p.views_dir / im["name"])),
                          cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = cv2.imread(str(p.views_masks_dir /
                              im["name"].replace(".jpg", ".png")),
                          cv2.IMREAD_GRAYSCALE) > 127
        mse = float(((pred - gt) ** 2)[mask].mean())
        psnrs.append(-10 * np.log10(mse))
    print(f"{ply_path}: masked PSNR over {n_views} views: "
          f"mean {np.mean(psnrs):.2f} dB  min {np.min(psnrs):.2f}  "
          f"max {np.max(psnrs):.2f}")
    return float(np.mean(psnrs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--n-views", type=int, default=12)
    a = ap.parse_args()
    run_eval(a.project, a.ply, a.n_views)


if __name__ == "__main__":
    main()
