"""Stage 5 (gsplat backend): floater-suppressing trainer on gsplat's public API.

Replaces nerfstudio/Splatfacto with a compact trainer adding the mechanisms of
the StableGS/TIDI-GS class using maintained tooling:
  - MCMC strategy with a hard splat cap (Quest budget enforced at training)
  - opacity regularization (floaters fade instead of surviving local minima)
  - sparse-depth supervision from our quality-filtered COLMAP points
  - mask-weighted loss (vehicle + sky excluded, as in the Splatfacto path)
Trains directly in the input (leveled global) frame: no export rotation.

Run inside cuda_env.bat with the training venv (needs compiled gsplat).
Data-loading and projection helpers are torch-free for testability.
"""
import argparse
import math
import time

import cv2
import numpy as np

from .colmap_export import load_images_txt, load_points3d_txt
from .io_utils import Project, ensure_dir, read_json
from .trajectory import quat_to_R


def load_cell_dataset(project, cell_id):
    """Read one cell's COLMAP model + images + masks into numpy arrays."""
    p = project if isinstance(project, Project) else Project(project)
    cdir = p.cells_dir / f"cell_{cell_id:03d}" / "colmap"
    meta = read_json(p.views_meta)
    K = np.array([[meta["fx"], 0, meta["cx"]],
                  [0, meta["fy"], meta["cy"]],
                  [0, 0, 1.0]])
    imgs_meta = load_images_txt(cdir / "images.txt")
    images, masks, w2c = [], [], []
    for im in imgs_meta:
        img = cv2.imread(str(p.views_dir / im["name"]))
        mask = cv2.imread(str(p.views_masks_dir /
                              im["name"].replace(".jpg", ".png")),
                          cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        masks.append((mask > 127).astype(np.uint8))
        qw, qx, qy, qz = im["q"]
        T = np.eye(4)
        T[:3, :3] = quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = im["t"]
        w2c.append(T)
    pts = load_points3d_txt(cdir / "points3D.txt")
    return {"images": np.stack(images), "masks": np.stack(masks),
            "w2c": np.stack(w2c), "K": K,
            "points": pts[:, :3].copy(),
            "rgbs": pts[:, 3:6].copy() / 255.0,
            "names": [im["name"] for im in imgs_meta]}


def project_depths(points, w2c, K, width, height, margin=2):
    """Per view: (u, v, depth) of COLMAP points that project inside the image.
    Sparse ground-truth depth for the depth loss."""
    out = []
    for T in w2c:
        cam = points @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        front = z > 0.5
        uv = cam[front] @ K.T
        u = uv[:, 0] / uv[:, 2]
        v = uv[:, 1] / uv[:, 2]
        ok = ((u >= margin) & (u < width - margin)
              & (v >= margin) & (v < height - margin))
        out.append((u[ok], v[ok], z[front][ok]))
    return out


def _ssim(a, b, C1=0.01 ** 2, C2=0.03 ** 2):
    """Mean SSIM over an 11x11 gaussian window; a, b: (H,W,3) float tensors."""
    import torch
    import torch.nn.functional as F
    win = torch.exp(-(torch.arange(11, device=a.device, dtype=a.dtype) - 5) ** 2
                    / (2 * 1.5 ** 2))
    win = (win / win.sum())
    kern = (win[:, None] @ win[None, :]).expand(3, 1, 11, 11)
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    mu_x = F.conv2d(x, kern, padding=5, groups=3)
    mu_y = F.conv2d(y, kern, padding=5, groups=3)
    sxx = F.conv2d(x * x, kern, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, kern, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, kern, padding=5, groups=3) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / \
        ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return s.mean()


def run_gstrain(project_dir, cell_id=0, iters=30000, cap=1_500_000,
                sh_degree=3, opacity_reg=0.01, scale_reg=0.01,
                depth_weight=0.05, seed=42):
    import torch
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"
    p = Project(project_dir)
    ds = load_cell_dataset(p, cell_id)
    n_img, H, W = ds["images"].shape[:3]
    print(f"dataset: {n_img} views {W}x{H}, {len(ds['points']):,} seed points")

    depths_per_view = project_depths(ds["points"], ds["w2c"], ds["K"], W, H)
    Ks = torch.tensor(ds["K"], device=device, dtype=torch.float32)
    w2c_all = torch.tensor(ds["w2c"], device=device, dtype=torch.float32)

    # --- parameters seeded from COLMAP points (gsplat reference init) ---
    pts = torch.tensor(ds["points"], device=device, dtype=torch.float32)
    rgbs = torch.tensor(ds["rgbs"], device=device, dtype=torch.float32)
    from scipy.spatial import cKDTree
    d3, _ = cKDTree(ds["points"]).query(ds["points"], k=4)
    avg_dist = torch.tensor(d3[:, 1:].mean(axis=1),
                            device=device, dtype=torch.float32)
    N = len(pts)
    means = torch.nn.Parameter(pts.clone())
    scales = torch.nn.Parameter(torch.log(avg_dist.clamp(1e-4)[:, None]
                                          .repeat(1, 3)))
    quats = torch.nn.Parameter(torch.rand(N, 4, device=device))
    opacities = torch.nn.Parameter(torch.logit(
        torch.full((N,), 0.1, device=device)))
    sh0 = torch.nn.Parameter(((rgbs - 0.5) / 0.28209479177387814)[:, None, :])
    shN = torch.nn.Parameter(torch.zeros(
        N, (sh_degree + 1) ** 2 - 1, 3, device=device))
    params = torch.nn.ParameterDict({
        "means": means, "scales": scales, "quats": quats,
        "opacities": opacities, "sh0": sh0, "shN": shN})

    scene_scale = float(np.linalg.norm(
        ds["w2c"][:, :3, 3].max(0) - ds["w2c"][:, :3, 3].min(0)))
    lr_cfg = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
              "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr,
                                        "name": k}], eps=1e-15)
                  for k, lr in lr_cfg.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / iters))

    strategy = MCMCStrategy(cap_max=cap, verbose=False)
    state = strategy.initialize_state()

    imgs_t = torch.tensor(ds["images"], device="cpu")   # uint8, stream per step
    masks_t = torch.tensor(ds["masks"], device="cpu")

    t0 = time.time()
    for step in range(iters):
        i = np.random.randint(n_img)
        gt = imgs_t[i].to(device).float() / 255.0
        m = masks_t[i].to(device).float()[..., None]
        sh_deg = min(step // 1000, sh_degree)

        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors, sh_degree=sh_deg,
            viewmats=w2c_all[i][None], Ks=Ks[None], width=W, height=H,
            render_mode="RGB+ED")
        rgb, depth = render[0, ..., :3], render[0, ..., 3]

        rgb_m, gt_m = rgb * m, gt * m
        l1 = (rgb_m - gt_m).abs().mean()
        ssim = 1.0 - _ssim(rgb_m, gt_m)
        loss = 0.8 * l1 + 0.2 * ssim
        loss = loss + opacity_reg * torch.sigmoid(params["opacities"]).mean()
        loss = loss + scale_reg * torch.exp(params["scales"]).mean()
        u, v, dgt = depths_per_view[i]
        if depth_weight > 0 and len(dgt):
            dsamp = depth[v.astype(int), u.astype(int)]
            dref = torch.tensor(dgt, device=device, dtype=torch.float32)
            loss = loss + depth_weight * (dsamp - dref).abs().mean()

        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info,
                                    lr=optimizers["means"].param_groups[0]["lr"])
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()

        if step % 1000 == 0 or step == iters - 1:
            n = len(params["means"])
            print(f"step {step:6d}  loss {loss.item():.4f}  "
                  f"splats {n:,}  {(time.time()-t0)/60:.1f} min", flush=True)

    export_ply(params, ensure_dir(p.export_dir) / f"cell_{cell_id:03d}.ply")


def export_ply(params, path):
    """Standard 3DGS PLY layout, positions in the training (global) frame."""
    import torch
    from plyfile import PlyData, PlyElement
    with torch.no_grad():
        means = params["means"].cpu().numpy()
        scales = params["scales"].cpu().numpy()
        quats = (params["quats"] /
                 params["quats"].norm(dim=-1, keepdim=True)).cpu().numpy()
        op = params["opacities"].cpu().numpy()
        sh0 = params["sh0"].cpu().numpy()
        shN = params["shN"].cpu().numpy()
    n = len(means)
    fields = ([("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
              + [(f"f_dc_{i}", "f4") for i in range(3)]
              + [(f"f_rest_{i}", "f4") for i in range(shN.shape[1] * 3)]
              + [("opacity", "f4")]
              + [(f"scale_{i}", "f4") for i in range(3)]
              + [(f"rot_{i}", "f4") for i in range(4)])
    v = np.zeros(n, dtype=fields)
    v["x"], v["y"], v["z"] = means.T
    for i in range(3):
        v[f"f_dc_{i}"] = sh0[:, 0, i]
    rest = shN.transpose(0, 2, 1).reshape(n, -1)   # channel-major like Inria
    for i in range(rest.shape[1]):
        v[f"f_rest_{i}"] = rest[:, i]
    v["opacity"] = op
    for i in range(3):
        v[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        v[f"rot_{i}"] = quats[:, i]
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))
    print(f"exported {n:,} splats -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--cell", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--cap", type=int, default=1_500_000)
    ap.add_argument("--depth-weight", type=float, default=0.05)
    ap.add_argument("--opacity-reg", type=float, default=0.01)
    a = ap.parse_args()
    run_gstrain(a.project, a.cell, a.iters, a.cap,
                depth_weight=a.depth_weight, opacity_reg=a.opacity_reg)


if __name__ == "__main__":
    main()
